---
name: video-analyzer
description: Use when editing raw video footage conversationally — transcribing, selecting cuts, building a Decision Matrix, and exporting an XML timeline for Resolve/Premiere/FCP. Triggers on requests to edit/cut a video file, build a trailer, or work with combined_analysis.json.
---

Read `SKILL.md` at the repo root completely before doing anything else. That
file (plus `skills/master_editor/SKILL.md` for the cut-decision phase) is the
full, current process for an editing session — do not read `main.py`,
`helpers/*.py`, or `visual_connector.py` to figure out how the pipeline
works; the CLI contract documented there is complete.
