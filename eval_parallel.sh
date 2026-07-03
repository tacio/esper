#!/usr/bin/env bash
# Parallel held-out evaluation over a directory of `.task` bundles.
#
# Each task is fully independent (its own operator fit + buffers), so we shard
# the task list across `nproc` worker processes — one `mojo run src/arc_solve.mojo`
# per shard — and aggregate the per-task results afterward. Near-linear speedup
# with zero shared state (no in-process threading: ESWorkspace scratch and the
# global RNG are not thread-safe). This is purely a faster harness around the
# same `--report` driver `run_tests.sh` / the docs use; the numbers are identical.
#
# Usage:
#   ./eval_parallel.sh <task_dir> [out_file] [n_workers] [fit_N fit_iters]
# e.g.
#   ./eval_parallel.sh data_bin/arc2_train scratch/arc2_train_results.txt
#   ./eval_parallel.sh data_bin/arc2_train scratch/arc2_train_v2.txt 16 64 1500
# The optional trailing pair is forwarded as `--fit N ITERS` (the documented
# corpus budget); omitted = the full FIT_N/FIT_ITERS proof budget.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TASK_DIR="${1:?usage: ./eval_parallel.sh <task_dir> [out_file] [n_workers] [fit_N fit_iters]}"
OUT_FILE="${2:-scratch/eval_results.txt}"
N="${3:-$(nproc)}"
FIT_ARGS=()
if [[ $# -ge 5 ]]; then FIT_ARGS=(--fit "$4" "$5"); fi

mapfile -t TASKS < <(ls "$TASK_DIR"/*.task 2>/dev/null | sort)
TOTAL="${#TASKS[@]}"
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No .task bundles found in $TASK_DIR" >&2
    exit 1
fi
# Don't spawn more workers than tasks.
if [[ "$N" -gt "$TOTAL" ]]; then N="$TOTAL"; fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Evaluating $TOTAL tasks across $N workers ($(nproc) cores)..."
START=$(date +%s)

# Round-robin the tasks into N shard arg-files, then launch one worker per shard.
for ((w = 0; w < N; w++)); do : >"$WORK/shard_$w.args"; done
for ((i = 0; i < TOTAL; i++)); do
    printf '%s\n' "${TASKS[$i]}" >>"$WORK/shard_$((i % N)).args"
done

pids=()
for ((w = 0; w < N; w++)); do
    # xargs feeds the shard's task paths as argv to one mojo invocation.
    (xargs -a "$WORK/shard_$w.args" -d '\n' \
        mojo run -I src src/arc_solve.mojo --report "${FIT_ARGS[@]}" \
        >"$WORK/out_$w.txt" 2>&1) &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

# Aggregate the per-task lines from every shard into the output file.
mkdir -p "$(dirname "$OUT_FILE")"
grep -h '^  task:' "$WORK"/out_*.txt | sort >"$OUT_FILE" || true

SCORED=$(wc -l <"$OUT_FILE")
END=$(date +%s)

# Global solved count (held-out is field 4) and mean held-out, recomputed across
# all shards (each shard's own "Solved X/Y" footer is per-shard, so ignore it).
awk -v total="$TOTAL" -v scored="$SCORED" -v secs="$((END - START))" '
{ ho = $4; sum += ho; if (ho >= 0.99) solved++ }
END {
    printf "--------------------------------------------------\n"
    printf "Solved %d / %d  (solve rate: %.3f%%, mean held-out: %.6f)\n", \
        solved, total, (total ? 100.0 * solved / total : 0), (scored ? sum / scored : 0)
    printf "scored %d tasks in %ds\n", scored, secs
}' "$OUT_FILE" | tee -a "$OUT_FILE"

if [[ "$fail" -ne 0 ]]; then
    echo "WARNING: a worker exited non-zero; results may be short of $TOTAL." >&2
fi
