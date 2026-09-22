# CLAUDE.md

Guidance for Claude Code when working in this bundle. This is the runtime
distribution of Video Analyzer for editor workstations - pipeline + editing
skill only, no dev/test/panel code. Read `SKILL.md` and
`skills/master_editor/SKILL.md` completely before an edit session.

## Product contract

Video Analyzer is a conversational editing skill, not a batch renderer. The
agent runs deterministic analysis, selects one versioned audio/visual
context, proposes a cut strategy, waits for approval, follows
`skills/master_editor/SKILL.md`, waits for Decision Matrix approval, and
delivers an XML timeline for a professional NLE.

Never cut inside a word, never write the EDL before the required approvals,
never mix artifacts from different runs.

## Storage is resolved, selected, and allocated

Use the storage CLI - never construct paths or IDs by hand:

```powershell
$workspace = python helpers/storage.py resolve '<video>'
$combined  = python helpers/storage.py select-combined '<video>' --run 3
$xmlDir    = python helpers/storage.py create-xml '<video>' --name '30 second trailer'
```

For normal footage the resolver returns `<video_parent>/<video_stem> edit`.
Pass the same `--output-root` override to every command if one is used.
`AI_EDITS_ROOT` in `.env` is the default output root on this machine.

Workspace layout:

```text
STATIC/                              reusable scene boundaries / images
ACTIVE/TRANSCRIPTION/<ID - label>/   transcript + raw + dynamics
ACTIVE/VISUAL CACHE/<ID - label>/    visual description cache
COMBINED ANALYSIS/<ID - label>/      primary JSON + run companions
XML/<ID - name>/                     matrix + EDL + timeline
```

One global sequence spans all artifact kinds. Complete manifests govern
selection; folder order and gaps do not. `project.md`, the Scene Map, and any
debug media stay beside their selected `combined_analysis.json`. The Decision
Matrix, `edl.json`, and `timeline.xml` stay together in the directory
`create-xml` returns.

## Pipeline commands

```powershell
python main.py '<video>' --language cs --asr-backend nemo
python main.py '<video>' --transcription-run 1 --visual-run 2 --asr-backend nemo
python main.py '<video>' --skip-json --combined-run 3
```

`--asr-backend nemo` + `NEMO_URL` pointed at the local Canary server is the
transcription path on this machine. Explicit selectors are
`--transcription-run`, `--visual-run`, `--combined-run`. `--language ""`
enables auto-detection.

Transcription only (scene extraction + ASR + audio dynamics, no vision, no
combined artifact):

```powershell
python main.py '<video>' --language cs --asr-backend nemo --transcription-only
```

Helpers:

```powershell
python helpers/canary_server.py --port 8002      # the local ASR server; start it first
python helpers/pack_transcripts.py --transcription-dir '<selected transcription run>'
python helpers/export_fcpxml.py '<XML session>/edl.json'
```

## Architecture

`main.py` is deterministic orchestration:

1. `helpers/transcribe.py` posts audio to the ASR server (`--asr-backend
   nemo` -> `NEMO_URL` -> local Canary) and writes processed + raw payloads
   to one transcription run.
2. `visual_connector.py::load_or_extract_scenes` owns `STATIC` (PySceneDetect
   boundaries + ffmpeg screenshots).
3. `visual_connector.py::describe_scenes` writes one visual-cache run through
   local Ollama vision (`OLLAMA_URL`, `OLLAMA_VISION_MODEL`).
4. `helpers/audio_analyzer.py` writes per-scene/speaker dynamics beside the
   selected transcript.
5. `expand_transcript_with_visuals` writes the scene-centric
   `combined_analysis.json` with producer provenance.
6. `analysis_report.json` stays in that combined run; a timing row appends to
   `analysis_runs.csv` (or `ANALYTICS_CSV`).

The agent is the editorial brain. `SKILL.md` owns conversation and strategy;
`skills/master_editor/SKILL.md` owns Walter Murch's Rule of Six, the 4-pass
workflow, split edits, and the Decision Matrix.

In `edl.json`, `sources` holds the original absolute media path; selected
debug media (if ever used) goes only in `source_overrides`.

## Services on this machine

- **Vision:** local Ollama (`OLLAMA_URL`, `OLLAMA_VISION_MODEL=gemma4:12b`).
  If unset, `visual_connector.py` falls back to a recorded dev-only mock.
- **Transcription:** local Canary ASR server (`helpers/canary_server.py`,
  port 8002), reached via `--asr-backend nemo` + `NEMO_URL`. ASR
  `nvidia/canary-1b-v2`; diarization `nvidia/diar_streaming_sortformer_4spk-v2`
  (CC-BY-4.0); `langid_ambernet` sub-window language routing with hysteresis
  thresholds (0.5 stay / 0.75 switch), 4s sub-windows, 2-sub-window
  agreement, 35s max ASR call. A combined MarbleNet + PANNs (`panns-inference`,
  installed alongside `nemo_toolkit`) VAD/music gate runs before LID on every
  sub-window. No `HF_TOKEN` needed. If `nemo_toolkit` is not
  installed the server returns a mock so the rest of the pipeline still runs.
  `main.py` auto-starts and auto-stops it (`helpers/local_service_autostart.py`,
  reusing `helpers/service_supervisor.py`) around each run that needs it -
  only for a local `NEMO_URL`, never a remote one. Cold model load is
  verified at 5+ minutes on this hardware, paid on every run since the
  server is stopped again afterward; start it by hand first (see
  `README.md`) and leave it running to skip that cost across several runs.
- `ffmpeg` / `ffprobe` are auto-provided by the `static-ffmpeg` package
  (`main.py` calls `static_ffmpeg.add_paths()` before importing anything
  that shells out to them) - no manual PATH setup needed, though either can
  still be installed by hand and will be used instead if already on PATH.

## Errors

Read the raised exception first - this code invests in specific errors
(`ArtifactNotFoundError`, source-identity mismatch messages in
`helpers/storage.py`). If a fix needs information only the operator has (a
model choice, a hardware constraint, confirmation to change default
behaviour), stop and ask for that one thing.

Verify a code change with:

```powershell
python -m py_compile main.py visual_connector.py helpers/*.py
```
