# Documentation

Two different readers use this folder, and they want different things.

**If you want to run Afterimage,** you need four of these documents. Start
with [USAGE.md](USAGE.md); the rest of the folder is the evidence behind the
claims, not instructions.

**If you want to check whether the claims are true,** the research record is
here in full, including the things that did not work.

## Using Afterimage

| Document | What it answers |
|---|---|
| [USAGE.md](USAGE.md) | How do I install it and generate text? |
| [CONFIGURATION.md](CONFIGURATION.md) | What knobs exist, and which models are supported? |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | It broke. Now what? |
| [FAQ.md](FAQ.md) | Short answers to the common questions. |

## Understanding it

| Document | What it answers |
|---|---|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | What each method actually does, measured next to AirLLM. Start here if you want the results narrative. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Where the code lives and what the module boundaries are. Start here if you want to change it. |

## The research record

Every performance claim in the top-level README traces into these. They are
kept because a claim you cannot check is not a result; the negative and
contradicted findings are here too, deliberately.

| Document | What it answers |
|---|---|
| [ALL_HYPOTHESES_AND_BASELINES.md](ALL_HYPOTHESES_AND_BASELINES.md) | **The controlling results table.** Every hypothesis H0-H18, every baseline, current verdicts. A test (`tests/test_measured_outcomes_match_docs.py`) enforces that the code registry cannot silently drift from this file. |
| [RESULTS_LOG.md](RESULTS_LOG.md) | The append-only run ledger every number traces back to, including regressions and corrections. Nothing is edited away. |
| [RESEARCH_METHODS.md](RESEARCH_METHODS.md) | The protocol: evidence levels L0-L3, per-hypothesis gates, and what counts as a kill criterion. |
| [HYPOTHESIS_LINEAGE.md](HYPOTHESIS_LINEAGE.md) | Where each idea came from, what was borrowed, and what was changed. No mechanism here is claimed as confirmed-novel. |
| [LITERATURE.md](LITERATURE.md) | The survey of prior work on running models larger than VRAM. |
| [REPRODUCE.md](REPRODUCE.md) | One command per reported number, and the environment facts a rerun will not match by default. |
| [CROSS_MODEL_BENCHMARK_2026-08-22.md](CROSS_MODEL_BENCHMARK_2026-08-22.md) | Do the Qwen3-14B conclusions transfer to another family and scale? Historical record of that campaign as run. |

## Forward research (not yet evidence)

| Document | What it answers |
|---|---|
| [SPECULATION_TREE_RESEARCH.md](SPECULATION_TREE_RESEARCH.md) | The next research line (H19-H34): candidate-tree speculation, its pre-registered gates, and what is implemented versus planned. The CPU-only offline analysis it describes is real and tested; **no GPU campaign has run for it yet**, and the document says so per hypothesis. |
| [SPECULATION_NOVELTY_LEDGER.md](SPECULATION_NOVELTY_LEDGER.md) | Closest known prior art for each idea in that line, with search dates. Nothing in it is asserted as established novelty. |

## What is deliberately not here

Some material stays on the author's disk rather than in the published
repository, because a clone should carry the engine and the evidence for its
claims, not someone's in-progress writing:

- `docs/archive/`: superseded planning documents, kept for traceability.
- `paper/`: manuscript drafts, outlines, and campaign planning.
- `scripts/local/`: personal session scripts and local experiment harnesses.
- `results/paper-comparison/`: live campaign output, including partial and
  exploratory attempts. Completed, paper-eligible artifacts are curated into
  the date-stamped published result set instead.

The rule: **measurements and the protocol that produced them are public;
writing about them is not.**
