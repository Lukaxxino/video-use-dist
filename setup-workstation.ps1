<#
.SYNOPSIS
  One-shot dependency install for a Video Analyzer editor workstation (Windows + NVIDIA GPU).

.DESCRIPTION
  Creates a .venv in this folder, installs a CUDA PyTorch, the core pipeline
  (pyproject.toml), and the local Canary ASR server dependencies, then runs a
  compile + import smoke check and seeds a local-only .env.

  Run it from inside this bundle folder; it treats its own directory as the root.

.PARAMETER TorchIndex
  PyTorch wheel index. Default cu128 (torch 2.8.0+cu128). Use cu126 / cu124
  for older drivers.

.PARAMETER SkipCanary
  Skip the heavy nemo_toolkit / Canary ASR server dependencies. Use on a
  machine that will only call a remote transcription endpoint.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File setup-workstation.ps1
#>
[CmdletBinding()]
param(
  [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
  [switch]$SkipCanary
)

$ErrorActionPreference = "Stop"

function Section($t) { Write-Host ""; Write-Host "=== $t ===" -ForegroundColor Cyan }
function Warn($t)    { Write-Host "WARNING: $t" -ForegroundColor Yellow }
function Have($cmd)  { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot"

# ---------------------------------------------------------------------------
Section "1/8  Preflight checks"
# ---------------------------------------------------------------------------
$py = if (Have "py") { "py -3" } elseif (Have "python") { "python" } else { $null }
if (-not $py) { throw "Python not found. Install Python 3.10-3.11 (x64) and re-run." }
$pyver = (& cmd /c "$py --version") 2>&1
Write-Host "Python: $pyver"
$verOk = & cmd /c "$py -c ""import sys; print(1 if sys.version_info[:2] >= (3,10) and sys.version_info[:2] <= (3,12) else 0)"""
if ($verOk.Trim() -ne "1") { Warn "Python 3.10-3.12 recommended (pyproject requires >=3.10; nemo_toolkit is happiest on 3.10/3.11)." }

if (Have "nvidia-smi") {
  $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) 2>&1
  Write-Host "GPU: $gpu"
} else {
  Warn "nvidia-smi not found. Transcription (Canary) needs a CUDA GPU. Vision via Ollama can still run CPU/GPU."
}

foreach ($c in "ffmpeg","ffprobe") {
  if (Have $c) { Write-Host "${c}: OK (already on PATH)" } else { Write-Host "${c}: will be auto-provided by the static-ffmpeg package on first pipeline run." }
}
if (Have "ollama") {
  Write-Host "ollama: OK"
} else {
  Write-Host "ollama not found -- installing via winget ..."
  try {
    winget install --id Ollama.Ollama --accept-package-agreements --accept-source-agreements --silent
    if (Have "ollama") { Write-Host "ollama: installed." } else { Warn "winget reported success but 'ollama' is still not on PATH -- open a new shell (PATH updates need a fresh process) or install manually from https://ollama.com." }
  } catch {
    Warn "Automatic Ollama install via winget failed ($_). Install manually from https://ollama.com for local vision."
  }
}
if (Have "claude") { Write-Host "claude (Claude Code): OK" } else { Warn "Claude Code CLI ('claude') not found. Install it per Anthropic's docs - it drives the editing skill." }
if (-not (Have "git")) { Warn "git not found." }

# ---------------------------------------------------------------------------
Section "2/8  Create virtual environment (.venv)"
# ---------------------------------------------------------------------------
$VenvDir = Join-Path $RepoRoot ".venv"
if (Test-Path (Join-Path $VenvDir "Scripts\python.exe")) {
  Write-Host ".venv already exists, reusing it."
} else {
  & cmd /c "$py -m venv "".venv"""
  if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
}
$VPy  = Join-Path $VenvDir "Scripts\python.exe"
$VPip = "$VPy -m pip"

# ---------------------------------------------------------------------------
Section "3/8  Upgrade pip / setuptools / wheel"
# ---------------------------------------------------------------------------
& cmd /c "$VPip install --upgrade pip setuptools wheel"
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

# ---------------------------------------------------------------------------
Section "4/8  Install CUDA PyTorch  ($TorchIndex)"
# ---------------------------------------------------------------------------
# Install torch FIRST from the CUDA index, then pin it with a constraints
# file so later installs (insanely-fast-whisper, nemo_toolkit) cannot silently
# swap in a driver-incompatible PyPI default build.
& cmd /c "$VPip install --index-url $TorchIndex torch torchaudio"
if ($LASTEXITCODE -ne 0) { throw "torch install failed. Try a different -TorchIndex (cu126 / cu124) for your driver." }

$Constraint = Join-Path $RepoRoot "torch-constraint.txt"
& cmd /c "$VPip freeze | findstr /R /I ""^torch"" > ""$Constraint"""
Write-Host "Wrote torch constraint:"
Get-Content $Constraint | ForEach-Object { Write-Host "  $_" }

# ---------------------------------------------------------------------------
Section "5/8  Install core pipeline (pyproject.toml)"
# ---------------------------------------------------------------------------
& cmd /c "$VPip install -c ""$Constraint"" ."
if ($LASTEXITCODE -ne 0) { throw "core pipeline install failed." }

# ---------------------------------------------------------------------------
Section "6/8  Install local Canary ASR server dependencies"
# ---------------------------------------------------------------------------
if ($SkipCanary) {
  Write-Host "Skipped (-SkipCanary). This machine must point NEMO_URL at a remote transcription endpoint."
} else {
  & cmd /c "$VPip install -c ""$Constraint"" ""nemo_toolkit[asr]>=3.0.0"" soundfile python-multipart pyyaml"
  if ($LASTEXITCODE -ne 0) { throw "nemo_toolkit install failed." }
}

# ---------------------------------------------------------------------------
Section "7/8  Smoke checks"
# ---------------------------------------------------------------------------
& cmd /c "$VPy -m py_compile main.py visual_connector.py helpers\*.py"
if ($LASTEXITCODE -ne 0) { throw "py_compile failed - the bundle is broken." }
Write-Host "py_compile: OK"

& cmd /c "$VPy -c ""import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"""

if (-not $SkipCanary) {
  & cmd /c "$VPy -c ""import nemo; print('nemo_toolkit', getattr(nemo,'__version__','?'))"""
}

& cmd /c "$VPy -c ""import requests, librosa, cv2, scenedetect, opentimelineio, fastapi, uvicorn, PIL, numpy; print('core imports: OK')"""
if ($LASTEXITCODE -ne 0) { throw "a core dependency failed to import." }

# ---------------------------------------------------------------------------
Section "8/8  Seed .env"
# ---------------------------------------------------------------------------
$EnvPath = Join-Path $RepoRoot ".env"
$Template = Join-Path $PSScriptRoot ".env.example"
if (Test-Path $EnvPath) {
  Write-Host ".env already exists - left untouched. Compare against .env.example if needed."
} else {
  Copy-Item $Template $EnvPath
  Write-Host "Copied .env.example -> .env  (local-only services). Edit it if any model runs elsewhere."
}

Write-Host ""
Write-Host "=== DONE. Next steps ===" -ForegroundColor Green
Write-Host "  1. ollama pull gemma4:12b        # or your standard vision tag"
Write-Host "  2. .venv\Scripts\python.exe helpers\canary_server.py --port 8002    # first run downloads the models"
Write-Host "  3. Smoke test the pipeline (transcript only, no vision/editor):"
Write-Host "       .venv\Scripts\python.exe main.py ""<some test clip>.mp4"" --language cs --asr-backend nemo --transcription-only"
Write-Host "  4. Full run once Ollama has the vision model:"
Write-Host "       .venv\Scripts\python.exe main.py ""<some test clip>.mp4"" --language cs --asr-backend nemo"
Write-Host "  5. Then open a Claude Code session in this repo and drive an editing session via the video-analyzer skill."
