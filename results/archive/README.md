# Archived judge scores

Superseded runs, kept for provenance and never loaded by `judge.load_cached`,
which reads only `results/judge_scores_<judge>.jsonl`.

## judge_scores_gemini_flash_reasoning_default.jsonl

The secondary judge scored with its provider-default reasoning budget: it
wrote out its deliberation before returning a score. Superseded because that
cost several hundred billed output tokens per call to receive one digit, and
because the hardest pairs overran the token budget mid-thought — every
unparseable reply in this project came from that, concentrated in the
`matched` condition, so the loss was not random.

The judge was re-run with `reasoning="none"`. These records are kept rather
than deleted so the switch is auditable: what was measured before the change
is on disk, not merely described.

**Not comparable to the active file as an ablation.** These were scored on
quoted interpretations while deliberating; the active file's stripped-quote
records were scored without deliberating. Two variables differ, so no
difference between them is attributable to either.
