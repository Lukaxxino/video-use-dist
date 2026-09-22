---
name: media-pool-curator
description: Use after a media_pool_edit strategy is confirmed to curate reusable, reviewable source moments for the DaVinci Resolve Media Pool without creating a timeline.
---

# Media Pool Curator

## Purpose and boundary

Turn one confirmed edit brief and one selected `combined_analysis.json` into a
quality-first collection of reusable source moments. The editor reviews the
collection and chooses which moments Resolve will materialize as Media Pool
subclips.

This skill selects moments; it does not call Resolve or write Resolve state.
Never allocate an XML session, build an edit order, create a Decision Matrix,
or write EDL/XML/timeline artifacts. Materialization belongs to the panel only
after explicit review confirmation.

## Inputs

Require all of these from the root skill or Orchestrator:

- the confirmed brief and editorial strategy;
- the exact selected combined-artifact path and ID;
- stable source keys and original absolute media paths;
- `max_selects` (default 200, allowed 1–500);
- whether handles are enabled and the requested before/after seconds.

The maximum is a ceiling, never a quota. Return fewer moments whenever that is
the stronger editorial result. The brief decides relevance; it must never
override source truth, legal boundaries, or technical quality evidence.

## Source access and authority

Use the selected combined artifact as the only authority. Use compact scene
scripts or ledgers for navigation, then retrieve the complete scene records for
every candidate before judging, changing, or rejecting it. A compact summary
is never evidence by itself.

Preserve stable source, combined-artifact, scene, and time provenance. IDs are
assigned by Python; never invent or rewrite IDs in model output.

## Four-pass workflow

Run the passes in this order. Persist each completed pass so an interrupted job
can resume without replaying accepted work.

### 1. Scout

Search broadly for moments useful under the brief. It may nominate at most
twice `max_selects`; this is only a search ceiling. Fetch complete details for
every serious candidate. Prefer complete dialogue thoughts, gestures,
reactions, actions, establishing shots, transitions, and ending beats over
arbitrary scene fragments.

Each candidate identifies the core source interval, primary category, proposed
secondary tags, brief relevance, technical evidence, and complete provenance.
Do not use handles to make a weak or incomplete core appear valid.

### 2. Serial Critic

Review candidates in adaptive batches while maintaining a compact ledger of
already accepted and rejected moments. Judge technical usability, brief value,
emotional or story value, clean natural boundaries, visual action, uniqueness,
and reuse potential. For every candidate return exactly one `keep`, `reject`,
or `revise` verdict with reasons. A revision may change only the candidate's
core or descriptive metadata and must retain its Python-owned identity and
provenance.

Commit each validated batch before continuing. After all batches, reconcile
duplicates and near-duplicates globally. If a compact ledger itself exceeds
the context budget, reconcile hierarchically. Before removing or changing an
item during reconciliation, retrieve its complete source detail; never decide
from ledger text alone.

### 3. Coverage Editor

Audit compact accepted/rejected ledgers against the brief and the whole source.
Look for meaningful gaps in people, speakers, locations, actions, emotions,
shot types, story functions, and editorial use cases. Coverage is not a quota:
do not add weak moments merely to fill a category.

Every proposed addition must be backed by complete source detail and must pass
the same Serial Critic before it can join the accepted set.

### 4. Final Curator

For every accepted item, work from complete detail and finalize its concise
title, literal description, practical use, core interval, one primary category,
secondary tags, quality score, relevance score, and selection reason. Then run
a global ordering and uniqueness reconciliation. Do not change inclusion or
boundaries from compact text alone.

Return no more than `max_selects`. Prefer fewer excellent, varied, immediately
usable choices over padding the result to the ceiling.

## Complete-request context budget

Every serialized model request must be estimated in full and remain at or
below 90,000 tokens. Count system instructions, tools, history, current full
records, ledgers, schema, and response allowance—not only the scene payload.

- Build adaptive batches from whole records. Never slice, abbreviate, or
  truncate an item to make a request fit.
- If one complete record cannot fit, stop with a specific context-budget error.
- If a response is truncated, discard that response and retry smaller batches.
  A single-record truncation is an explicit error, never partial success.
- Compact ledgers may guide navigation and comparison, but no final decision
  may rely on them without retrieving complete detail.

## Moment and taxonomy rules

One physical source moment becomes exactly one select. Give it exactly one
primary category; express other plausible uses as normalized secondary tags.
Do not duplicate the interval under multiple bins or labels. Reject moments
that are redundant, technically unusable, irrelevant to the brief, or unable
to stand as a coherent editorial unit.

The fixed primary categories are:

- `hook`
- `story-dialogue`
- `emotion-reaction`
- `b-roll`
- `transition-establishing`
- `ending-button`

Add at most four custom primary categories when the brief genuinely requires
them. Custom labels must be distinct from fixed categories and safe as Resolve
bin names. Categories organize; they do not impose minimum counts.

## Boundaries and handles

The model proposes only the semantic core interval. Dialogue edges must be on
word boundaries; visual edges must remain within the authoritative scene and
source. Python validates non-overlap and converts the interval to inclusive
constant-frame-rate frame bounds.

Optional before/after handles are editor settings, disabled by default. Python
applies, clamps, and outward-snaps them after approval; handles never change
the model's core judgment. Variable-frame-rate sources are rejected unless an
explicitly normalized constant-frame-rate source is provided.

## Review handoff

Produce a review collection grouped by primary category. Every item must show
its short title, literal description of what happens, practical editorial use,
core timing, and secondary tags. All items start checked. The editor may toggle
whole groups or individual items and may adjust handle settings.

Stop after presenting the validated review collection. Nothing enters Resolve
until the editor confirms that checklist. The Orchestrator then materializes
only checked items and owns journaling, batching, verification, and recovery.

## Failure rules

Fail explicitly rather than returning a partial or ambiguous collection when:

- the selected artifact or provenance changes;
- full detail required for a decision is unavailable;
- a request or response cannot fit without truncation;
- a model omits a candidate or returns unknown/duplicate IDs;
- taxonomy, boundary, overlap, count, or constant-frame-rate validation fails.

Do not silently skip failed records and do not substitute timeline output.
