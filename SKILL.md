---
name: video-analyzer
description: Use when a user wants a conversational analysis that produces either an XML timeline edit or a reviewed collection of Resolve Media Pool selects.
---

# Video Analyzer

## Principle

1. Reason from the selected `combined_analysis.json`: word-level transcript,
   scene descriptions, speakers, and audio dynamics belong together.
2. Ask → confirm → execute → iterate → persist. Do not make cut artifacts
   until the user confirms the plain-English strategy.
3. Do not assume a format or genre. Inspect the material first.
4. Stop at the output appropriate to the selected mode. Timeline edits stop at
   XML; Media Pool edits stop at a confirmed select collection for the Resolve
   panel. This pipeline does not render a finished video.

## Hard rules

1. Never cut inside a word. Snap every edge to a word boundary in the selected
   combined artifact.
2. Pad each cut edge for ASR timestamp drift. Use roughly 30–200 ms, adjusted
   to the material.
3. Use word-level verbatim ASR, not SRT or phrase-level timestamps.
4. Stop and ask `Do you agree with this cut strategy?` before invoking the
   mode-specific editor skill.
5. Resolve and select storage through `helpers/storage.py`. Never invent an
   artifact directory or ID.
6. Keep user artifacts in the resolved video workspace. The shared benchmark
   `analysis_runs.csv` beside `main.py` is the only project-root output
   exception.
7. Treat `main.py`, every file in `helpers/`, and `visual_connector.py` as a
   black box during an editing session. The CLI flags and workspace contract
   documented here and in `skills/master_editor/SKILL.md` are the complete
   interface — do not open them to understand how the pipeline works.
   Debugging or extending the pipeline itself is a different task: see
   `README.md`'s Architecture reference and your agent guide (`CLAUDE.md` /
   `AGENTS.md`)'s error-handling protocol for that.

## System at a glance

Every file in this system, one line each — this is the complete
interface; opening these to understand behavior is unnecessary:

| File | Role |
|---|---|
| `main.py` | Orchestrates the pipeline: reserves/selects each versioned artifact and calls the producers below in order. |
| `helpers/storage.py` | Resolves/reserves/selects every versioned artifact path and ID. Always go through this, never build a path by hand. |
| `helpers/transcribe.py` | ASR client; on this bundle only `--asr-backend nemo` is operable (points at the local Canary server below) - no `whisper` server is bundled, so always pass `--asr-backend nemo`. |
| `visual_connector.py` | PySceneDetect scene extraction + Ollama vision descriptions. |
| `helpers/audio_analyzer.py` | Per-scene/speaker loudness and words-per-second dynamics. |
| `helpers/debug_video.py` | Burns scene/transcript overlays into a debug MP4. |
| `helpers/export_fcpxml.py` | Converts `edl.json` into `timeline.xml`. |
| `helpers/pack_transcripts.py` | Reads one transcription run's `transcript.json` into a phrase-level `takes_packed.md`. |
| `helpers/canary_server.py` | The bundled local ASR+diarization server (`nvidia/canary-1b-v2` + streaming Sortformer), reached via `--asr-backend nemo` + `NEMO_URL`; no `HF_TOKEN` needed. See `CLAUDE.md` for the tuning details. |

## Canonical storage

Never derive the workspace from `video.parent`. Ask the storage helper:

```powershell
$workspace = python helpers/storage.py resolve '<video>'
```

It always returns `<AI_EDITS_ROOT>/<video_stem> edit` - every edit file on
this machine lives under `AI_EDITS_ROOT` (set in `.env`), never beside the
source media. If `AI_EDITS_ROOT` is not configured it raises instead of
guessing. `--output-root <path>` is available on `main.py` and every storage
helper command when an explicit override is required; pass the same override
to every command of the session.

The resolved workspace has this shape:

```text
<resolved workspace>/
├── STATIC/
│   ├── manifest.json
│   ├── scenes.json
│   └── scenes/
├── ACTIVE/
│   ├── TRANSCRIPTION/<ID - label>/
│   └── VISUAL CACHE/<ID - label>/
├── COMBINED ANALYSIS/<ID - label>/
│   ├── manifest.json
│   ├── combined_analysis.json
│   ├── analysis_report.json
│   ├── scene_script.txt
│   ├── project.md
│   ├── scene_map.md
│   └── debug-*.mp4
├── XML/<ID - name>/
    ├── edit_decision_matrix.md
    ├── edl.json
    └── timeline.xml
└── MEDIA POOL SELECTS/<ID - name>/
    ├── manifest.json
    └── selects.json
```

IDs share one workspace-wide sequence. Gaps between artifact kinds are normal.
Do not choose a run by scanning directory names.

## Process

### 0. Convert MXF/MOV/MPX sources to an MP4 proxy

If the source is `.mxf`, `.mov`, or `.mpx`, convert it before anything else,
without asking - the pipeline analyzes the proxy, not the original. The proxy
goes inside the source's own workspace, keeping the source's file stem so it
resolves to the same workspace:

```powershell
$workspace = python helpers/storage.py resolve '<source>'
# proxy: "$workspace\PROXY\<source stem>.mp4"
```

Use `ffmpeg`/`ffprobe` from PATH, or `.venv\Scripts\static_ffmpeg.exe` /
`static_ffprobe.exe` if they're not there. First list the audio streams:

```powershell
ffprobe -v error -select_streams a -show_entries stream=index,channels -of csv=p=0 '<source>'
```

Pick the audio explicitly - never `-map 0:a?`, which copies every track and
the pipeline then transcribes only the first one:
- one audio stream: `-map 0:a:0`
- several mono streams (typical broadcast MXF, tracks 1+2 = program stereo):
  `-filter_complex "[0:a:0][0:a:1]amerge=inputs=2[a]" -map "[a]"`

```powershell
ffmpeg -hide_banner -y -i '<source>' -map 0:v:0 <audio map from above> `
  -vf "yadif=deint=interlaced,scale=-2:'min(720,ih)'" `
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p `
  -c:a aac -b:a 192k -ac 2 -movflags +faststart '<proxy>'
```

Keep the source frame rate (no `-r`) so proxy timecodes map 1:1 to the
original. Use `libx264`, not NVENC (NVENC failed on a workstation driver
older than the one current ffmpeg builds need). Tell the operator in one line
which audio tracks went into the proxy, so they can ask for a different pair.

Pass the proxy path to `main.py`. The proxy exists only so analysis can
extract screenshots and audio - it is deleted as soon as analysis succeeds
(step 1). In `edl.json`, `sources` holds the original `.mxf`/`.mov` absolute
path - the NLE edits the original, never the proxy.

### 1. Establish language and analyze

Transcription defaults to Czech. If the source language is known, pass its ISO
639-1 code; use an empty value for auto-detection:

```powershell
python main.py '<video>' --language cs
python main.py '<video>' --language en
python main.py '<video>' --language ''
```

A normal run creates new versioned transcription, visual-cache, and combined
artifacts. To reuse a stage, select it explicitly:

```powershell
python main.py '<video>' --transcription-run 1 --visual-run 2
```

The legacy skip switches mean “select the newest compatible complete run.”
Prefer explicit IDs when the user names a run.

If the user only wants transcripts (no editing session), `--transcription-only`
stops after ASR and audio dynamics and skips visual analysis and the combined
artifact — see `CLAUDE.md`'s Pipeline commands section.

If step 0 made a proxy, delete it once `main.py` has finished successfully
(`select-combined` returns a `combined_analysis.json`); keep it only while a
run failed and will be retried. Everything the session needs afterwards -
screenshots, transcript, dynamics, combined analysis - is already in the
workspace. From then on pass the original source path to every storage
command: the workspace depends only on the file name, which the original and
the proxy share, and these commands don't read the video file itself.

Deleting the proxy has one consequence: burned-in debug media (step 6) and
`main.py` runs that reuse this analysis's runs need the exact proxy file
(they check its size and hash), and a re-encoded proxy won't match. If the
operator asks for either later, say it takes a fresh proxy and a fresh
analysis.

### 2. Select the Primary View before pre-scan

Always select one complete combined artifact before reading footage context.
Use the user's requested run; otherwise omit `--run` to select the newest
complete run:

```powershell
$combined = python helpers/storage.py select-combined '<video>' --run 3
$combinedDir = Split-Path -Parent $combined
```

If an output override was used for analysis, pass the same override:

```powershell
$combined = python helpers/storage.py select-combined '<video>' --run 3 --output-root '<workspace>'
```

Read only `$combined` for the initial scene-centric pre-scan. Do not silently
switch to another run later in the session.

### 3. Resume memory and converse

`project.md` and `scene_script.txt` both belong beside the selected combined
artifact. They answer different questions — do not read one expecting the
other:

```powershell
$projectLog = Join-Path $combinedDir 'project.md'
$sceneScript = Join-Path $combinedDir 'scene_script.txt'
```

`project.md` is conversational memory: a session log you append to by hand
(step 7). It only exists once a prior session has actually reached step 7, so
expect it to be absent on a video's first session. If it exists, summarize
the last session in one sentence.

`scene_script.txt` is the shoot/video metadata source, not `project.md` —
`main.py`'s pipeline (`helpers/scene_script.py`) writes it automatically for
every combined run, CLI or Resolve-panel-triggered, so it always exists once
analysis has completed. Its first lines, if any were supplied at analysis
time (the panel's Create Task form, or `main.py`'s `--director`/`--project`/
`--notes`/`--recorded-date` flags), are a small header — `# VIDEO`,
`# RECORDED`, `# DIRECTOR`, `# PROJECT`, `# NOTES` — read straight out of
`combined_analysis.json`'s own small `metadata` object. Read just that
header, not the rest of the file, to open the conversation already knowing
the material's director/project/recording date instead of asking for
anything the pipeline already recorded; a field left blank at analysis time
is simply absent from the header, not an error.

Then describe the material and collect target length, pacing, must-keep
moments, and must-cut moments.

**When the user is still choosing a direction** (e.g. "plan a promo/trailer",
"what would make a good teaser") rather than handing you a fixed brief, lay
the options out before proposing one cut strategy:

1. Say what data you actually have. If a stage is thin or missing (e.g. no
   usable visual descriptions), name it and adapt — do not pretend it's
   there.
2. A plain-language synopsis of what actually happens — the characters, the
   situation, the turns — written from the transcript and scene data so a
   reader who has not seen the footage follows the story. Not a scene dump.
   Then the distinct threads as a numbered one-sentence list, and a brief
   callout of any ready-made assets (a built-in voiceover line, a title
   card, a recurring motif).
3. Establish the target audience / channel / brand / use for the piece. If
   the user hasn't said, ask once — the ranking that follows is "most
   compelling *to that audience*", and that framing is what makes the
   recommendation sharp instead of a neutral list.
4. A ranked options table: each candidate direction, why it pulls that
   audience, the ready-made lines/moments it would use with their source
   timecodes, and any risk. Then one clear recommendation with a one-line
   reason.
5. For the chosen direction, a beat sheet with rough timecodes and, when
   narration carries it, the voiceover treatment (which lines, in what
   order, against what).

This is still conversation — it produces the plain-English strategy that
step 4 then stops and confirms.

### 4. Propose and confirm the strategy

Give a 4–8 sentence cut direction shaped by the selected material. Stop and
ask exactly: `Do you agree with this cut strategy?`

### 5. Route by output mode

After strategy confirmation, route exactly once:

- `timeline_edit` → `skills/master_editor/SKILL.md`
- `media_pool_edit` → `skills/media-pool-curator/SKILL.md`

The output mode is fixed when the task is created. Never reinterpret a Media
Pool request as a timeline request or allocate artifacts from the other mode.

For `media_pool_edit`, pass the confirmed brief and strategy, exact combined
artifact path and ID, stable source map, maximum-select ceiling, and optional
handle settings to the Media Pool Curator. It returns one validated review
collection with every item checked by default. Stop for checklist confirmation;
the panel materializes only checked items as Resolve subclips. This path never
creates a Decision Matrix, XML session, EDL, XML, or timeline order.

For `timeline_edit`, continue through the Master Editor as follows.

### 5a. Execute timeline edits through the Master Editor

After confirmation, read and follow `skills/master_editor/SKILL.md`. Pass it:

- the original absolute source-video path;
- the exact combined path returned by `select-combined`;
- the selected combined run ID;
- any debug-media path chosen inside `$combinedDir`;
- the requested edit-session name.

The Master Editor allocates an ID-bearing XML session, writes its Decision
Matrix there, stops for approval, and only then writes the EDL and XML.

This strategy-confirmation step — one plain-English strategy, one
`Do you agree with this cut strategy?` stop-and-ask — happens exactly once
regardless of what the Master Editor does with it next. In an ordinary CLI
session the confirmed strategy feeds one Master Editor edit session, ending
in one Decision Matrix, one approval, one `edl.json`, and one `timeline.xml`,
per `skills/master_editor/SKILL.md` section 7. When this session was
explicitly started by the Orchestrator acting as the Resolve panel's task
coordinator to run a generation job, the same confirmed strategy instead
feeds the Master Editor's panel generation mode
(`skills/master_editor/SKILL.md` section 8): the same strategy and source
context drive exactly three independent proposal runs, each with its own
Decision Matrix, EDL, and AI-written description, and `timeline.xml` is
written later, only for whichever one variant the user approves. Root-skill
behavior does not otherwise change between these two cases — only the Master
Editor's internal fan-out and its XML timing change, and only when the
Orchestrator explicitly requested it.

### 6. Optional versioned debug media

Debug generation must name the combined run. This reuses run `003` without
rerunning transcription, extraction, vision, or audio analysis:

```powershell
python main.py '<video>' --skip-json --combined-run 3 --burn-scenes --burn-transcript
$combined = python helpers/storage.py select-combined '<video>' --run 3
$combinedDir = Split-Path -Parent $combined
$debugMedia = Join-Path $combinedDir 'debug-scenes-transcript.mp4'
```

Other mode-specific names are `debug-scenes.mp4` and
`debug-transcript.mp4`. Never look for `<video>_debug.mp4` beside the source.
For an `.mxf`/`.mov`/`.mpx` source whose proxy was already deleted (step 1),
this isn't available without a fresh proxy and analysis.

### 7. Iterate and persist

On feedback, revise the strategy and create a new XML session through the
Master Editor. Do not overwrite a previous session. Append this shape to the
selected combined artifact's `project.md`:

```markdown
## Session N — YYYY-MM-DD

**Combined artifact:** ID and absolute `combined_analysis.json` path
**XML session:** ID and absolute session path
**Strategy:** one paragraph
**Decisions:** take choices and reasons
**Outstanding:** deferred items
```

## Optional transcript packing

Pack one selected transcription run, never a directory of mixed runs:

```powershell
python helpers/pack_transcripts.py --transcription-dir '<workspace>/ACTIVE/TRANSCRIPTION/001 - whisper'
```

The helper reads only `transcript.json` and writes `takes_packed.md` in that
same run. The selected combined artifact remains the authoritative editorial
view.

## Anti-patterns

- Reading a “latest-looking” directory instead of calling `select-combined`.
- Mixing a Scene Map, project log, or debug media from another combined run.
- Writing the Decision Matrix before allocating an XML session.
- Reusing an XML directory for a later edit.
- Putting a debug path in `sources`; original media belongs in `sources`, and
  selected debug media belongs in `source_overrides`.
- Cutting from audio gaps without checking visual action or reactions.
- Re-transcribing when a compatible complete transcription run is selected.
