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
function Test-RealPython($cmd) {
  # The Microsoft Store alias (WindowsApps\python.exe) satisfies Get-Command
  # but only points at the Store; a real interpreter prints "Python 3.x".
  # stderr is merged inside cmd so PowerShell 5.1 doesn't turn it into a
  # terminating error under $ErrorActionPreference = "Stop".
  try { $out = (& cmd /c "$cmd --version 2>&1") | Out-String } catch { return $false }
  return ($LASTEXITCODE -eq 0 -and $out -match "Python 3\.")
}
$py = $null
foreach ($cand in "py -3", "python") {
  if ((Have $cand.Split(" ")[0]) -and (Test-RealPython $cand)) { $py = $cand; break }
}
if (-not $py) {
  Write-Host "No working Python (only the Microsoft Store alias, or nothing) -- installing Python 3.11 for this user via winget ..."
  try {
    winget install --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements --silent
  } catch {
    Warn "winget Python install failed ($_)."
  }
  # PATH changes need a new process; put the per-user install first for this one.
  $userPyDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311"
  if (Test-Path (Join-Path $userPyDir "python.exe")) {
    $env:PATH = "$userPyDir;$userPyDir\Scripts;$env:PATH"
    if (Test-RealPython "python") { $py = "python" }
  }
  if (-not $py) { throw "Python not found. Install Python 3.11 (x64): 'winget install Python.Python.3.11 --scope user' or python.org, then re-run." }
}
$pyver = (& cmd /c "$py --version 2>&1")
Write-Host "Python: $pyver"
$verOk = & cmd /c "$py -c ""import sys; print(1 if sys.version_info[:2] >= (3,10) and sys.version_info[:2] <= (3,12) else 0)"""
if ($verOk.Trim() -ne "1") { Warn "Python 3.10-3.12 recommended (pyproject requires >=3.10; nemo_toolkit is happiest on 3.10/3.11)." }

if (Have "nvidia-smi") {
  $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) 2>&1
  Write-Host "GPU: $gpu"
} else {
  Warn "nvidia-smi not found. Transcription (Canary) needs a CUDA GPU. Vision via Ollama can still run CPU/GPU."
}

if ((Have "ffmpeg") -and (Have "ffprobe")) {
  Write-Host "ffmpeg/ffprobe: OK (already on PATH)"
} else {
  # static-ffmpeg can still provide it on first pipeline run, but its download
  # failed behind a TLS-inspecting proxy (then WinError 2 at extraction), and
  # SKILL.md step 0 calls ffmpeg directly for MXF/MOV proxies -- so install a
  # system ffmpeg as the backup.
  Write-Host "ffmpeg not on PATH -- installing via winget as a backup to static-ffmpeg ..."
  try {
    winget install --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements --silent
    if (Have "ffmpeg") { Write-Host "ffmpeg: installed." } else { Warn "ffmpeg installed but not on PATH in this shell yet -- a new shell will see it; static-ffmpeg covers this run." }
  } catch {
    Warn "winget ffmpeg install failed ($_); static-ffmpeg will try on first pipeline run."
  }
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
# --use-feature=truststore: verify TLS against the Windows certificate store
# (where a corporate TLS-inspection root CA lives) instead of pip's bundled
# certifi; a fresh venv's pip otherwise fails with CERTIFICATE_VERIFY_FAILED
# behind such a proxy. pip >= 24.2 does this by default once upgraded.
& cmd /c "$VPip install --use-feature=truststore --upgrade pip setuptools wheel"
if ($LASTEXITCODE -ne 0) {
  Warn "pip upgrade with truststore failed; retrying without it."
  & cmd /c "$VPip install --upgrade pip setuptools wheel"
  if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (behind a TLS-inspecting proxy? see the workstation-setup skill's known failures)." }
}

# Same fix for everything else that runs in this venv (requests, httpx,
# huggingface_hub model downloads, static-ffmpeg): a .pth file makes every
# interpreter start inject truststore. Certificate verification stays on --
# it just trusts the Windows store, which also holds the public roots, so
# this is harmless on networks without TLS inspection.
& cmd /c "$VPip install truststore"
if ($LASTEXITCODE -ne 0) { throw "truststore install failed." }
$SitePackages = (& $VPy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
Set-Content -Path (Join-Path $SitePackages "zz_truststore_windows.pth") -Value "import truststore; truststore.inject_into_ssl()" -Encoding ascii
Write-Host "truststore: TLS in this venv verifies against the Windows certificate store."

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
  & cmd /c "$VPip install -c ""$Constraint"" ""nemo_toolkit[asr]>=3.0.0"" soundfile python-multipart pyyaml panns-inference"
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
  # Pre-fetch panns' labels CSV + Cnn14 checkpoint (~300 MB) the same way the
  # runtime does: importing panns_inference otherwise shells out to `wget`,
  # which Windows doesn't have, and then fails on the missing CSV.
  & cmd /c "$VPy -c ""import sys; sys.path.insert(0, 'helpers'); from nemo_vad_worker import _ensure_panns_assets; _ensure_panns_assets(None); import panns_inference; print('panns_inference: OK')"""
  if ($LASTEXITCODE -ne 0) { throw "panns_inference failed to import - the Canary VAD/music gate needs it on every real transcription, not just at install time." }
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
