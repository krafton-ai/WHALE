# Running the experiments

Twelve runs: three domains times four conditions.

| Condition | Paper name | What alternates |
|---|---|---|
| `rsft_only` | weight-only baseline | weights only; the harness stays at `h0` |
| `mh_only` | harness-only baseline | the harness only; the model stays at `theta0` |
| `whale` | WHALE | both, on a fixed per-cycle budget `(E, I)` |
| `adaptive_whale` | adaptive WHALE | both, each phase ended by a patience rule |

Each launcher lives in `domains/<domain>/run/`. They are readable wrappers: they
set the documented variables and call the training driver and the harness-search
module. Every setting can be overridden from the environment.

---

## 1. Prerequisites

1. Prepare the datasets for the domain — see [data.md](data.md).
2. Install the vendored framework and its dependencies (each domain ships its
   own copy under `domains/<domain>/verl/`):

   ```bash
   cd domains/<domain>
   pip install -e verl
   pip install -r requirements.txt        # chess_puzzles only
   ```

3. Export the credentials in [§4](#4-credentials).
4. Start the domain's service, if it has one:

   | Domain | Service | Command |
   |---|---|---|
   | Search QA | retrieval server over the FAISS index | `python search_r1/search/retrieval_server_async.py` |
   | Mathematical Reasoning | code-execution sandbox | `bash scripts/start_retool_cpu_sandbox.sh` |
   | Chess Puzzles | none | — |

   Search QA expects the retriever endpoints in `SEARCH_R1_RETRIEVAL_URLS`.

---

## 2. The four conditions

All commands are run from the repository root.

### Weight-only baseline

```bash
domains/search_qa/run/rsft_only.sh
domains/math_reasoning/run/rsft_only.sh
domains/chess_puzzles/run/rsft_only.sh
```

Trains under the fixed base harness for the domain's full epoch budget (4, 6 and
4 epochs). Override `TRAINER_TOTAL_STEPS` to change the budget and
`HARNESS_PATH` to train under a different harness.

### Harness-only baseline

```bash
ITERATIONS=40 domains/search_qa/run/mh_only.sh
ITERATIONS=60 domains/math_reasoning/run/mh_only.sh
ITERATIONS=40 domains/chess_puzzles/run/mh_only.sh
```

Searches over the frozen base model. Those iteration counts are the ones used in
the paper; `PROPOSALS_PER_ITER` defaults to the `M = 3` of the paper.

### WHALE

Defaults reproduce the main comparison. Note that the launcher defaults differ
from the bare defaults of the internal scripts, which sit at the `(0.2, 2)`
ablation point; the paper's runs passed the values below explicitly.

```bash
domains/search_qa/run/whale.sh
domains/math_reasoning/run/whale.sh
domains/chess_puzzles/run/whale.sh
```

Defaults are the main comparison's `(E, I) = (0.6, 6)`: `ROUND_STEPS` is 0.6
epoch of weight updates and `ROUND_MH_ITERS=5` harness-search iterations. `I = 6`
in the paper counts the evaluation of the incoming harness as iteration 1, which
is why the launcher passes 5.

To reproduce the schedule ablation, set the pair explicitly, for example

```bash
ROUND_STEPS=15 ROUND_MH_ITERS=1 domains/search_qa/run/whale.sh   # (0.2, 2)
ROUND_STEPS=15 ROUND_MH_ITERS=5 domains/search_qa/run/whale.sh   # (0.2, 6)
ROUND_STEPS=44 ROUND_MH_ITERS=1 domains/search_qa/run/whale.sh   # (0.6, 2)
ROUND_STEPS=74 ROUND_MH_ITERS=9 domains/search_qa/run/whale.sh   # (1.0, 10)
```

with one epoch equal to 74 steps in Search QA, 70 in Mathematical Reasoning and
64 in Chess Puzzles at the shared prompt batch size of 256.

### Adaptive WHALE

```bash
domains/search_qa/run/adaptive_whale.sh
domains/math_reasoning/run/adaptive_whale.sh
```

No `(E, I)` to tune. The weight-update phase runs at least `ROUND_STEPS` (0.2
epoch) and stops when its training reward, averaged over a sliding window of the
same length, has not improved for that many steps. The harness-search phase runs
at least `MH_MIN_ITERS` and stops after `MH_PATIENCE` iterations without a new
best training score, capped at `MH_MAX_ITERS`. The paper evaluates this condition
in Search QA and Mathematical Reasoning; the Chess Puzzles launcher is present
and functional but was not part of the reported results.

---

## 3. Where the results land

```
domains/<domain>/outputs/<RUN_NAME>/          weight-update checkpoints, alternation.log
domains/<domain>/meta_harness/runs/<mh-run>/  candidate harnesses, scores, accepted_harness.txt
```

`alternation.log` records, per cycle, the target step of the weight-update phase
and the harness accepted by the harness-search phase. A run resumes if you
relaunch the same command: a cycle whose checkpoint already exists is skipped,
and a harness-search phase that already recorded an accepted harness is reused.

---

## 4. Credentials

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | the harness-search proposer sessions |
| `WANDB_API_KEY` | training and evaluation logging; leave unset for console-only |
| `LLM_JUDGE_KEY_FILE` or `OPENAI_API_KEY` | the Search QA answer judge |

No credential is stored in this repository.

---

## 5. Adapting to your cluster

The launchers assume one machine with the GPUs visible to the process. For a
scheduler, wrap them; `domains/math_reasoning/scripts/slurm_sandbox_fusion_cpu.sh`
is a worked SLURM example, with `--partition=CPU_PARTITION` left as a placeholder
to fill in. Useful knobs:

| Variable | Meaning |
|---|---|
| `N_GPUS_PER_NODE` | GPUs used by the training phase |
| `CKPT_ROOT` | where checkpoints are written; defaults to `domains/<domain>/outputs` |
| `BASE_MODEL` | base policy; `Qwen/Qwen3.5-2B` for Search QA and Mathematical Reasoning, `Qwen/Qwen3.5-4B` for Chess Puzzles |
| `BASE_HARNESS` | the harness `h0` a run starts from; each domain's default is `environments/<env>/base_harness.py` |
| `MAX_ROUNDS` | safety cap on the number of cycles |

The paper's runs used eight H200s per node. The launchers here are cleaned
re-implementations of the scheduler scripts that produced those numbers. They
pass the same contract to the two operators — the weight-update phase receives
`trainer.total_training_steps` and the `trainer.online_rsft.*` overrides, and the
harness-search phase receives `BASELINE_HARNESS_OVERRIDE` (the incoming harness)
and `VLLM_MODEL` (the checkpoint just produced) — while node staging, retrieval
index pre-fetch and inter-phase GPU cleanup were dropped as site-specific.

The alternation driver was verified with the training and harness-search steps
replaced by stubs: cycle boundaries, the hand-off of the accepted harness into
the next weight-update phase, the hand-off of the new checkpoint into the next
harness search, the Hydra overrides, the adaptive flags, and resumption of a
half-finished run all behave as intended. An end-to-end GPU run of these
launchers has not been performed; the numbers in the paper come from the
scheduler scripts they were derived from.
