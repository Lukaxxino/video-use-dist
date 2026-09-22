---
name: scene-mapper
description: Use when the user explicitly asks for a complete Scene Map of a selected combined video-analysis run.
---

# Scene Mapper

## Status

This is opt-in. Do not invoke it in the normal Orchestrator or Master Editor
flow unless the user asks for a Scene Map by name.

## Principle

1. Never skip a scene. Walk the selected `combined_analysis.json` from first
   scene to last.
2. Invent editorial Roles for this specific video; do not apply a fixed
   taxonomy.
3. Carry forward verbatim dialogue, the full visual description, and all audio
   dynamics. Drop only per-word timestamps and logprobs.
4. Never modify `combined_analysis.json`. `scene_map.md` is a companion file
   inside the same combined artifact.

## Select one combined artifact

Do not calculate a workspace, search for a latest-looking folder, or reuse a
Scene Map from another run. Resolve and select with the storage helper:

```powershell
$workspace = python helpers/storage.py resolve '<video>'
$combined = python helpers/storage.py select-combined '<video>' --run 3
$combinedDir = Split-Path -Parent $combined
$sceneMap = Join-Path $combinedDir 'scene_map.md'
```

Omit `--run` only when the user wants the newest complete combined artifact.
If analysis used `--output-root`, pass that same override to both commands.
Footage directly inside `Test Videos/Tested Videos` resolves through the
helper to `Test Videos/Finished analysis/<video_stem> edit`; never assemble
that special path manually.

The input and output contract is:

```text
<one COMBINED ANALYSIS/ID - label>/
├── combined_analysis.json  ← read
└── scene_map.md            ← reuse or write
```

If `$sceneMap` exists, reuse it only because it is beside the exact `$combined`
returned above. Report the selected combined run and stop.

## Procedure

1. Read `$combined`. For long footage, read in chunks until the whole
   `scenes[]` array is covered. Record the highest `scene_number`.
2. For every scene in order, write exactly one block to `$sceneMap`:
   - `## Scene N [start_time-end_time] speakers: <speaker numbers>`
   - `Role:` your editorial judgment for this scene in this video
   - `Text:` verbatim `word`/`audio_event` content with normal spacing
   - `Visual:` every `visual_description` field, verbatim
   - `Audio:` one line per speaker from `audio_dynamics`
   - `Flags:` technical/editorial notes, or `-`
3. Verify the block count equals the input scene count, the first and last
   scene numbers match, and every input `scene_number` appears exactly once.
4. Report the total and the absolute `$sceneMap` path to the Orchestrator.

## Hard rules

- One block for every input scene, in the same order.
- Scene-level `start_time`/`end_time` only; no per-word timestamps or
  logprobs in the map.
- No paraphrasing or truncating dialogue or visual-description content.
- `Role` is invented per video. `Flags` is free text or `-`.
- Read and write inside the same selected combined artifact.
- Never write inside the code repository. The shared project-root analytics
  CSV is unrelated and remains the sole documented project-root output
  exception.

## Example block

```markdown
## Scene 14 [91.36-96.82] speakers: 0,2
Role: Genre mock
Text: "Vždycky nějakej pupkatej strejda si sedne, votevře lahváče a začne vzpomínat."
Visual: environment: interior living room; subjects: older man on sofa;
action: opens beer, leans back, gestures while reminiscing;
flaws_or_notes: none
Audio: speaker_0 pacing=Normal loudness=Normal
Flags: -
```

## Handoff

The only output is `$sceneMap`. Do not propose a cut strategy or edit the EDL.
The Orchestrator and Master Editor continue from this map plus the same
selected `$combined`.
