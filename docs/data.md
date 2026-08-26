# Datasets

Every domain uses three disjoint roles, matching the paper:

| Role | Symbol | What it feeds |
|---|---|---|
| Weight-update training | `D_weight` | the RSFT rollouts and the SFT step over verifier-accepted trajectories |
| Harness-search training | `D_harness` | scoring candidate harnesses inside Meta-Harness |
| Test | `D_test` | the reported `mean@8` accuracy; never seen during training or search |

`D_harness` holds 256 examples in every domain, and every candidate harness is
evaluated with one rollout per example, so a harness-search iteration costs
`256 x M` rollouts, where `M = 3` is the number of candidates proposed per
iteration.

Nothing below is committed to this repository; the commands regenerate it.

---

## 1. Search QA

**Source.** The Search-R1 question-answering mixture, built from the public
Natural Questions, TriviaQA, PopQA, HotpotQA, 2WikiMultihopQA, MuSiQue and
Bamboogle datasets, plus the 2018 Wikipedia dump used as the retrieval corpus.

### 1.1 QA splits

```bash
cd domains/search_qa
python scripts/data_process/qa_search_train_merge.py \
    --local_dir data/nq_hotpotqa_train --data_sources nq,hotpotqa
python scripts/data_process/qa_search_test_merge.py \
    --local_dir data/nq_hotpotqa_train \
    --data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
```

- **`D_weight`** — the merged `nq,hotpotqa` train file, 18,946 questions.
- **`D_test`** — the merged seven-dataset test file. The paper reports each of
  the seven benchmarks separately and their average.

### 1.2 Retrieval corpus and index

```bash
python scripts/download.py --save_path data/wiki-18
cat data/wiki-18/part_* > data/wiki-18/e5_Flat.index
gzip -d data/wiki-18/wiki-18.jsonl.gz
```

This is a flat FAISS index over `intfloat/e5-base-v2` embeddings of the 2018
Wikipedia dump, roughly 75 GB. `search_r1/search/index_builder.py` rebuilds it
from scratch if you prefer; `search_r1/search/retrieval_server_async.py` serves
it during training and harness search.

### 1.3 Harness-search and evaluation subsets

```bash
python scripts/data_process/build_meta_harness_nq_256.py     # D_harness, 256 examples
python scripts/data_process/build_nq_eval_256.py             # 256-question NQ eval
python scripts/data_process/build_test_100_no_tool.py        # 100-question smoke test
python scripts/data_process/build_final_test_excluding_val.py
python scripts/data_process/build_meta_harness_eval.py
```

`data/searchR1_processed/mh_dist_align_256.parquet` is `D_harness` and is the
default `EVAL_DATASET_PATH` of every alternating run.

---

## 2. Mathematical Reasoning

**Source.** DAPO-Math-17K for training, AIME 2024 and AIME 2025 for testing.

- **`D_weight`** — `data/retool/dapo_math_17k_dedup.parquet`, the de-duplicated
  DAPO-Math-17K split, 17,917 problems.
- **`D_harness`** — `data/retool/meta_harness_dapo17k_dedup_seed42_256.parquet`,
  256 problems sampled from the same pool with seed 42.
- **`D_test`** — AIME 2024 (`Maxwell-Jia/AIME_2024`) and AIME 2025
  (`yentinglin/aime_2025`) from the Hugging Face Hub, 30 problems each.

Download the two source datasets from the Hub, de-duplicate DAPO-Math-17K, and
write both parquet files under `domains/math_reasoning/data/retool/`. The
harness-search subset must be drawn from the training pool, not from AIME, so
that the test sets stay unseen.

Helper builders used while preparing SFT-style variants live in `scripts/`:
`build_sft_from_mh_trajectories.py` and `build_sft_from_rollout_jsonl.py`.

---

## 3. Chess Puzzles

**Source.** The Lichess open puzzle database.

```bash
cd domains/chess_puzzles
python scripts/prepare_chess_puzzle_data.py --output-dir data/chess_puzzle
```

The builder streams the database, keeps puzzles whose solution is between
`--min-solution-plies 1` and `--max-solution-plies 9`, and writes three parquet
files with `--seed 42`:

| File | Default size | Role |
|---|---|---|
| `lichess_puzzles_train_16384.parquet` | `--train-size 16384` | `D_weight` |
| `lichess_puzzles_mh_val_256.parquet` | `--mh-val-size 256` | `D_harness` |
| `lichess_puzzles_test_1024.parquet` | `--test-size 1024` | pool for `D_test` |

The paper evaluates on 256 test puzzles drawn from the test pool. The three
splits are disjoint by construction.

---

## 4. The base harness h0

Every condition in a domain starts from the same `h0`, at
`domains/<domain>/environments/<env>/base_harness.py`. For Search QA it forwards
the model's query unchanged, returns a single passage truncated to 200 tokens,
and terminates after two assistant turns; the shared retrieval and prompt
machinery it builds on lives beside it in `search_r1_env.py`.

## 5. Verifiers

Each domain scores a trajectory with a binary verifier `R(x, tau)` that is fixed
across the whole study; it is used identically for accepting RSFT trajectories,
for scoring harness candidates, and for test evaluation.

| Domain | Verifier |
|---|---|
| Search QA | answer extracted from `<answer>` tags, then judged against the reference answers by `gpt-5.4-mini` at temperature 0 (`verl/utils/reward_score/llm_judge.py`) |
| Mathematical Reasoning | strict `\boxed{...}` match against the ground truth, following DAPO |
| Chess Puzzles | one UCI move parsed and checked for legality with `python-chess`, then compared with the reference move; the reward is 1 only when the whole reference sequence is played |
