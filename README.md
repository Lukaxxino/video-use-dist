# Video Analyzer - editor workstation bundle

Runtime distribution for editor machines: the analysis pipeline + the
conversational editing skill. No dev, test, DaVinci-panel, or curator code.
Everything runs locally on this box - vision on Ollama (gemma4), transcription
on a local Canary ASR server.

## What's in here

```
main.py, visual_connector.py    pipeline entry points
helpers/*.py                    runtime helpers only (+ the 3 Canary subprocess workers)
pyproject.toml                  dependency list for `pip install .`
setup-workstation.ps1           one-shot installer
.env.example                    local-only config template
SKILL.md, skills/, CLAUDE.md    the editing skill Claude Code follows
.claude/skills/video-analyzer/  skill registration
```

## Install (fresh Windows machine, NVIDIA GPU)

### 1. Prerequisites (install by hand, once)

- Recent NVIDIA driver (`nvidia-smi` works).
- **Python 3.11 x64** (3.10-3.12 accepted).
- **Claude Code CLI** - install per Anthropic's docs; check `claude --version`.

`ffmpeg`/`ffprobe` and **Ollama** no longer need a manual install step:
`setup-workstation.ps1` installs Ollama via `winget` if it is not already on
PATH, and `ffmpeg`/`ffprobe` are auto-provided at first pipeline run by the
`static-ffmpeg` package (downloads a static build once, caches it, no
separate step). Either can still be installed by hand first if preferred -
both are detected and left alone.

### 2. Run the installer

From this folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup-workstation.ps1
```

It creates `.venv` here, installs Ollama via `winget` if missing, installs a
CUDA PyTorch (cu128 by default; pass `-TorchIndex
https://download.pytorch.org/whl/cu126` for an older driver) and pins it,
installs the core pipeline (`pip install .`, which pulls in `static-ffmpeg`
for `ffmpeg`/`ffprobe`) and the Canary ASR server deps (`nemo_toolkit[asr]` +
soundfile/python-multipart/pyyaml/panns-inference), runs a compile check, and seeds `.env`
from `.env.example`.

Flags: `-SkipCanary` (this box will call a remote transcript endpoint
instead), `-SkipTests`.

### 3. Pull the vision model

```powershell
ollama pull gemma4:12b
```

### 4. The local Canary ASR server starts itself - no manual step needed

`main.py` auto-starts `helpers\canary_server.py` on `NEMO_URL`'s port
(`.env` already points it at `http://localhost:8002/v1/transcribe`) whenever
`--asr-backend nemo` needs it and nothing is already listening there, and
stops it again once the run finishes. **This means every single run pays a
cold-load penalty - verified directly on this hardware at 5+ minutes** (Canary
+ Sortformer + LID + VAD model loading, not the actual transcription, which
is fast once warm) before the first byte of a response. If you are about to
run several clips back-to-back, start it once yourself and leave it running
instead - `main.py` detects and reuses an already-running server rather than
starting a second one:

```powershell
.venv\Scripts\python.exe helpers\canary_server.py --port 8002
```

First start downloads `nvidia/canary-1b-v2` + `nvidia/diar_streaming_sortformer_4spk-v2`.

### 5. Smoke test

```powershell
# transcript only (ffmpeg + Canary, no vision)
.venv\Scripts\python.exe main.py "<clip>.mp4" --language cs --asr-backend nemo --transcription-only

# full analysis (needs Ollama + the vision model)
.venv\Scripts\python.exe main.py "<clip>.mp4" --language cs --asr-backend nemo
```

Then open a Claude Code session in this folder and drive an editing session
through the `video-analyzer` skill to a `timeline.xml`.

## Notes

- **Transcription** = local Canary only. ASR `nvidia/canary-1b-v2`;
  diarization `nvidia/diar_streaming_sortformer_4spk-v2` (CC-BY-4.0,
  commercial use OK). No `HF_TOKEN`. The LID routing is the tuned build
  (0.5 stay / 0.75 switch confidence, 4s sub-windows, 2-sub-window agreement,
  35s max ASR call) matching the GH200 devcontainer.
- **Vision** = local Ollama, `gemma4:12b` (README compat: on Ollama 0.31.x,
  `gemma3:4b` emits garbage vision tokens; `gemma4:12b` and `llava:7b` work).
- **Portability:** no machine-specific paths in the code; all config is in
  `.env`. Artifact identity is content-based (`filename + size + sha1`), so a
  whole `<...> edit` workspace directory can be copied to another machine as
  long as the source media file is present there and `AI_EDITS_ROOT` /
  `--output-root` point at it.
- No music, grade, or titles - the pipeline stops at an editable XML
  timeline for Resolve / Premiere / FCP.
