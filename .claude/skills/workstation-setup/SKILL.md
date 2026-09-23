---
name: workstation-setup
description: Use when a fresh editor-workstation machine has this bundle but has not yet been set up - no working .venv, unsure whether Ollama/CUDA torch/nemo_toolkit/panns-inference/the local Canary server are installed - before starting any editing session on this box.
---

# Workstation Setup

## Overview

This bundle needs Python deps, a CUDA-matched PyTorch, Ollama + a vision
model, and the local Canary ASR server (`nemo_toolkit` + `panns-inference`)
before any editing session can run. `setup-workstation.ps1` already does
the deterministic install work and is the one source of truth for it - this
skill drives that script instead of reimplementing its steps, adds the
judgment a static script can't (reading a real failure and reacting to it),
and turns the operator's job from "read README.md and run five commands by
hand" into one conversation.

**Not in scope:** it does not duplicate `setup-workstation.ps1`'s install
logic, does not touch `main.py`/the pipeline, and does not decide whether
`gemma4:12b` fits this machine's VRAM beyond reporting what `nvidia-smi`
shows - if that reported VRAM looks clearly too small for the pull about
to happen, treat it as an operator-decision case (step 2) and ask rather
than proceeding and letting the pull fail.

## When to use

- First run on a machine: no `.venv` here yet, or `.env` is missing.
- Re-run after a machine's install broke or drifted (missing model, stale
  torch build) - anything that isn't just "run an editing session."

**Not for:** routine editing work once the machine is set up - that's the
`video-analyzer` skill.

## Flow

1. **Preflight (read-only).** Check `python`/`py` version, `nvidia-smi` +
   GPU/driver, `claude --version`, `git`. Report what you found before
   touching anything.
2. **Present the plan, get exactly one confirmation for the whole flow.**
   The single yes/no you ask for here covers everything through hand-off
   (step 8) - the `setup-workstation.ps1` run AND the post-install actions
   in step 6 (`ollama pull`, starting `canary_server.py` for its first,
   long weight download, the smoke test). Don't ask again before step 6
   just because it's a separate section of this document. State: what
   `setup-workstation.ps1` will do (its 8 sections, in order 1 of 8
   through 8 of 8), that steps 4 and 6 follow immediately after with no
   further prompt beyond the one-time edit-workspace question, the rough
   download size (CUDA torch + nemo_toolkit +
   Canary/Sortformer weights + Ollama + the vision model - tens of GB
   combined), and what it changes on the system (installs Ollama via
   `winget` if missing, creates `.venv` inside this folder - nothing else
   outside the repo). Mention that if this is a first install, you'll ask
   one more question right after the script finishes: where to store edit
   workspaces (step 4). Wait for a yes.

   After that yes, run to completion without further stops, EXCEPT for
   these three cases, which always require asking first even mid-flow:
   - a decision only the operator can make (an unfamiliar driver's
     `-TorchIndex`, whether reported VRAM looks too small for the vision
     model about to be pulled);
   - killing, restarting, or replacing a process that turns out to
     already be running (see step 6);
   - a failure that isn't in the table below (step 5) - a *table-listed*
     failure does not need a stop; apply its Fix and narrate what you're
     doing as you go, you don't need to ask permission for a fix that's
     already been documented and approved by a prior run.
3. **Run `setup-workstation.ps1`.** Stream its output rather than
   summarizing it after the fact - the operator should see progress as it
   happens. If Claude Code blocks `powershell -ExecutionPolicy Bypass -File
   ...` for you, don't work around it: ask the operator to run it
   themselves by typing `! powershell -ExecutionPolicy Bypass -File
   setup-workstation.ps1` in the prompt (the `!` runs it in this session,
   so its output lands in the conversation), then continue from its output.
4. **Confirm the edit-workspace location - only if `.env` was just
   created** (section 8 of 8 seeds it from `.env.example` only when no
   `.env` exists yet; if `.env` already existed, the script left it alone
   - skip this step entirely, don't ask). Ask where edit workspaces
   (`AI_EDITS_ROOT`) should live: the default is local disk
   (`%USERPROFILE%\Documents\AI-edits`, already in the seeded `.env`), or
   the operator may want a shared/network drive instead - see
   `helpers/storage.py`'s `get_env("AI_EDITS_ROOT")` for how it's consumed
   and the `ai-edits-workspace-output-root` context if this session has
   it. If
   they want the default, leave `.env` as-is. If they name a different
   path, update the `AI_EDITS_ROOT` line in `.env` to it and create that
   directory if it doesn't exist yet. This is a one-time question, not a
   pattern to repeat elsewhere in this flow.
5. **On failure, debug before retrying.** Read the actual error text
   first. Check the table below for a known cause - match on the
   underlying error, not the exact surrounding circumstance (e.g. the
   `panns_inference` row applies whether the import fails during setup's
   own smoke check or later on a real transcription; same error, same
   fix). If it's a known cause, apply its Fix directly and continue (no
   need to stop and ask - see step 2). If it's new, apply
   `superpowers:systematic-debugging` rather than guessing - find the root
   cause, then fix it, then re-run only the failed section rather than the
   whole script if that's safe, then add a row to the table. **Never kill
   or restart a process (Ollama, a stuck `canary_server.py`, an old
   install) without asking first**, even though the rest of this flow runs
   unattended.
6. **Post-install.** No separate confirmation needed (step 2 already
   covers this). `ollama pull gemma4:12b`; then `helpers\canary_server.py
   --port 8002`: first check whether something is already listening on
   that port - if so, don't touch it, just verify it's actually a Canary
   server answering the expected contract and treat that as satisfying
   "started"; if nothing is listening, start it fresh (first run
   downloads model weights - can take a while, let it finish; this is a
   *new* process, not a kill/restart, so it doesn't need a separate ask).
   If a process IS already running but looks like the wrong thing (wrong
   port owner, doesn't answer the expected contract), that's a
   kill/restart/replace decision - stop and ask before touching it. Then
   run the transcription-only smoke test from `README.md` against a clip
   the operator points to; confirm the result is real output, not the
   `"mock": true` fallback.
7. **Log every step as it happens** to `install-log.md` in this folder
   (create it if absent). Log at sub-step granularity, not just once per
   numbered Flow step above: each script section as it completes, the
   edit-workspace location decision (default kept, or the path chosen and
   why), the exact text of any failure, the fix applied and why, each
   post-install action (`ollama pull` done, canary_server started/found
   running, smoke-test result), and the final hand-off - each as its own
   timestamped entry, written when that thing happens, not batched into a
   summary at the end.
8. **Hand off.** Tell the operator the machine is ready and that the next
   conversation should go through the `video-analyzer` skill.

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: panns_inference` (or the VAD-gate subprocess fails) - whether this shows up during setup's own section 7 of 8 smoke check, or later on a real transcription after setup reported success | `panns-inference` wasn't installed alongside `nemo_toolkit` | Already fixed in this bundle's `setup-workstation.ps1` (installs `panns-inference` and smoke-checks the import at setup time) - if it recurs, the install was interrupted or edited; re-run section 6 of 8 by hand: `pip install -c torch-constraint.txt "nemo_toolkit[asr]>=3.0.0" soundfile python-multipart pyyaml panns-inference`, then re-run the section 7 of 8 smoke check |
| `ollama` still not on PATH right after the installer runs it via `winget` | PATH updates need a fresh shell/process | Open a new terminal (don't kill the current one on a guess it's stuck) and re-check `ollama --version` |
| `torch install failed` | CUDA index doesn't match this GPU's driver | This is an operator-decision case (see step 2), not a silent retry: ask for the driver version (`nvidia-smi`), then re-run with `-TorchIndex https://download.pytorch.org/whl/cu126` (or `cu124`) instead of guessing |
| Smoke test returns `"mock": true` | `nemo_toolkit` import failed silently, or `NEMO_URL` doesn't point at a running local Canary server | Re-run the `import nemo` smoke check from section 7 of 8 by hand; confirm `helpers\canary_server.py` is actually running on the port in `.env`'s `NEMO_URL` |
| `'wget' is not recognized` then `FileNotFoundError ...panns_data\class_labels_indices.csv` | Importing `panns_inference` downloads its assets via `wget`, which Windows lacks | The section 7 of 8 smoke check now pre-fetches them with `_ensure_panns_assets` first; if it recurs, run that smoke-check line by hand |
| pip `CERTIFICATE_VERIFY_FAILED` | Corporate TLS inspection: the company root CA is in the Windows store, pip trusts only `certifi` | The installer uses `--use-feature=truststore` and writes `zz_truststore_windows.pth` into the venv; if it still fails, the root CA is missing from the Windows store - that's for the operator's IT |
| `static-ffmpeg` SSL error, then `WinError 2` when extracting ffmpeg | Same TLS inspection | Fixed by the truststore `.pth`; the installer also installs `Gyan.FFmpeg` via winget as a backup |
| `ASR server returned 500`, server log shows `httpx.ConnectError` | Hugging Face model download blocked by TLS inspection | Check `zz_truststore_windows.pth` exists in `.venv\Lib\site-packages`; re-run section 3 of 8 if not |
| `ASR server returned 500`, server log shows `WinError 206` (only on long videos) | Per-window worker arguments overflowed Windows' 32767-char command line | Fixed (`--window-list`); if it recurs, this bundle is older than that fix - pull the latest |
| LID/diarization very slow; `torch.cuda.is_available()` is False inside the workers | Workers pinned `CUDA_VISIBLE_DEVICES=1` (a GH200 leftover) on a single-GPU box | Fixed (defaults to GPU 0); if still on CPU, check nothing sets `CUDA_VISIBLE_DEVICES` globally |
| Transcript far shorter than expected, a repeated-word loop ("keska keska..."), `silence` events over real speech | VAD gate reading the wrong MarbleNet class | Fixed on this bundle (class 1); before changing it again, follow the silence/speech check in `helpers\nemo_vad_worker.py`'s comment |
| `python` exists but `python --version` fails / opens the Microsoft Store | Only the Store alias, no real Python | The installer detects this and installs Python 3.11 per-user via winget |

When you find and fix a new one, add a row here - this table is how the
next operator's install goes faster than this one's.

## Quick reference

```powershell
powershell -ExecutionPolicy Bypass -File setup-workstation.ps1
ollama pull gemma4:12b
.venv\Scripts\python.exe helpers\canary_server.py --port 8002
.venv\Scripts\python.exe main.py "<clip>.mp4" --language cs --asr-backend nemo --transcription-only
```

`README.md` has the full manual walkthrough this skill automates.
`CLAUDE.md` has the pipeline architecture once the machine is set up.
