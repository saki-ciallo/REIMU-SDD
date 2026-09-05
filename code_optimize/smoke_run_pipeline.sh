#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMMON_CONFIG="$SCRIPT_DIR/configs/common.yaml"
SMOKE_CONFIG="$SCRIPT_DIR/configs/smoke_pipeline.yaml"
OUTPUT_ROOT="$SCRIPT_DIR/outputs"
ASV21_DATASETS="df"
CHECKPOINT_MODE="last"
MAX_TEST_SAMPLES=32
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: ./code_optimize/smoke_run_pipeline.sh [options]

Runs a minimal CUDA train -> test -> score flow for the optimized ADD stack.

Options:
  --output-root PATH       Output root, relative to code_optimize/ or absolute.
                           Default: code_optimize/outputs
  --datasets la|df|both    ASVspoof2021 track(s) to test and score.
                           Default: df (the short prefix contains both labels).
  --checkpoint-mode MODE   best, last, or both. Default: last.
  --max-test-samples N     Maximum samples per test dataset. Default: 32.
  --dry-run                Print commands without running them.
  -h, --help               Show this help.

The smoke profile intentionally uses a linear frontend, one Attention+MLP
block, CrossEntropyLoss, two optimizer steps, and torch_compile=false.
EOF
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    echo "Missing value for $1." >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --output-root)
      require_value "$1" "${2-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --datasets)
      require_value "$1" "${2-}"
      ASV21_DATASETS="$2"
      shift 2
      ;;
    --checkpoint-mode)
      require_value "$1" "${2-}"
      CHECKPOINT_MODE="$2"
      shift 2
      ;;
    --max-test-samples)
      require_value "$1" "${2-}"
      MAX_TEST_SAMPLES="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$ASV21_DATASETS" in
  la|df|both) ;;
  *) echo "--datasets must be la, df, or both; got: $ASV21_DATASETS" >&2; exit 2 ;;
esac
case "$CHECKPOINT_MODE" in
  best|last|both) ;;
  *) echo "--checkpoint-mode must be best, last, or both; got: $CHECKPOINT_MODE" >&2; exit 2 ;;
esac
if ! [[ "$MAX_TEST_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-test-samples must be a positive integer; got: $MAX_TEST_SAMPLES" >&2
  exit 2
fi

if [[ "$OUTPUT_ROOT" = /* ]]; then
  OUTPUT_ROOT_ABS="$OUTPUT_ROOT"
else
  OUTPUT_ROOT_ABS="$SCRIPT_DIR/$OUTPUT_ROOT"
fi
OUTPUT_ROOT_ABS="$(python - "$OUTPUT_ROOT_ABS" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
if [[ "$OUTPUT_ROOT_ABS" != "$SCRIPT_DIR" && "$OUTPUT_ROOT_ABS" != "$SCRIPT_DIR/"* ]]; then
  echo "--output-root must be inside $SCRIPT_DIR; got: $OUTPUT_ROOT_ABS" >&2
  exit 2
fi

for required_file in "$COMMON_CONFIG" "$SMOKE_CONFIG"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing configuration: $required_file" >&2
    exit 1
  fi
done

DATASETS_ROOT="$PROJECT_ROOT/../datasets"
if [[ ! -d "$DATASETS_ROOT/ASVspoof2019_16k_4s_fixed" ]]; then
  echo "Missing ASVspoof2019 dataset: $DATASETS_ROOT/ASVspoof2019_16k_4s_fixed" >&2
  exit 1
fi
if [[ "$ASV21_DATASETS" == "la" || "$ASV21_DATASETS" == "both" ]]; then
  if [[ ! -d "$DATASETS_ROOT/ASVspoof2021_LA_16k_4s_fixed" ]]; then
    echo "Missing ASVspoof2021 LA dataset: $DATASETS_ROOT/ASVspoof2021_LA_16k_4s_fixed" >&2
    exit 1
  fi
fi
if [[ "$ASV21_DATASETS" == "df" || "$ASV21_DATASETS" == "both" ]]; then
  if [[ ! -d "$DATASETS_ROOT/ASVspoof2021_DF_16k_4s_fixed" ]]; then
    echo "Missing ASVspoof2021 DF dataset: $DATASETS_ROOT/ASVspoof2021_DF_16k_4s_fixed" >&2
    exit 1
  fi
fi

if [[ ! -f "$PROJECT_ROOT/official_scores/ASVspoof2019_LA/ASVspoof2019.LA.cm.eval.trl.txt" ]]; then
  echo "Missing ASVspoof2019 official score assets under $PROJECT_ROOT/official_scores." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_ROOT/official_scores/2021/eval-package/main.py" ]]; then
  echo "Missing ASVspoof2021 eval package under $PROJECT_ROOT/official_scores." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH-}"
export PYTHONDONTWRITEBYTECODE=1

TEMP_CONFIG="$(mktemp "${TMPDIR-/tmp}/add-smoke-config.XXXXXX.yaml")"
cleanup() {
  rm -f "$TEMP_CONFIG"
}
trap cleanup EXIT

python - "$SMOKE_CONFIG" "$TEMP_CONFIG" "$OUTPUT_ROOT_ABS" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

import yaml

source_path, destination_path, output_root = sys.argv[1:]
with Path(source_path).open(encoding="utf-8") as handle:
    values = yaml.safe_load(handle)
if not isinstance(values, dict):
    raise TypeError(f"Smoke config must be a mapping: {source_path}")
values = deepcopy(values)
run = values.setdefault("run", {})
if not isinstance(run, dict):
    raise TypeError("Smoke config run section must be a mapping.")
run["output_root"] = output_root
with Path(destination_path).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(values, handle, allow_unicode=True, sort_keys=False)
PY

RUN_DIR="$(python - "$SCRIPT_DIR" "$COMMON_CONFIG" "$TEMP_CONFIG" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from add_system.configuration.experiments import resolve_experiment

print(resolve_experiment(sys.argv[2:]).build_settings().output_dir.resolve())
PY
)"

prepare_score_fixtures() {
  local tested_results_dir="$RUN_DIR/tested_results"
  local asv21_keys_dir="$PROJECT_ROOT/official_scores/2021/eval-package/keys"
  local asv19_protocol="$PROJECT_ROOT/official_scores/ASVspoof2019_LA/ASVspoof2019.LA.cm.eval.trl.txt"
  local tracks="$ASV21_DATASETS"

  echo
  echo "[prepare score fixtures]"
  printf '  '
  printf '%q ' python - "$tested_results_dir" "$asv19_protocol" "$asv21_keys_dir" "$tracks"
  printf '\n'
  if [[ "$DRY_RUN" == true ]]; then
    return
  fi

  # Official scorers require one score for every protocol row. The inference
  # stage stays small; these fixtures fill only the scorer input for the smoke.
  python - "$tested_results_dir" "$asv19_protocol" "$asv21_keys_dir" "$tracks" <<'PY'
from pathlib import Path
import sys

import pandas as pd

tested_results_dir = Path(sys.argv[1])
asv19_protocol_path = Path(sys.argv[2])
asv21_keys_dir = Path(sys.argv[3])
tracks = ["la", "df"] if sys.argv[4] == "both" else [sys.argv[4]]
neutral_score = 0.5


def select_prediction(prefix: str) -> Path:
    root_prediction = tested_results_dir / f"{prefix}root.csv"
    if root_prediction.is_file():
        return root_prediction
    candidates = sorted(
        path
        for path in tested_results_dir.glob(f"{prefix}*.csv")
        if not path.stem.endswith("_smoke_fixture")
    )
    if not candidates:
        raise FileNotFoundError(f"No prediction CSV found for {prefix!r} in {tested_results_dir}")
    return candidates[-1]


def score_map(prediction_path: Path) -> dict[str, float]:
    prediction = pd.read_csv(prediction_path, usecols=["utterance_id", "bonafide_score"])
    if prediction["utterance_id"].duplicated().any():
        raise ValueError(f"Duplicate utterance_id in {prediction_path}")
    return {
        str(utterance_id): float(score)
        for utterance_id, score in prediction.itertuples(index=False, name=None)
    }


def fallback_score(row_number: int) -> float:
    # Keep the fixture a soft-score file even inside small protocol groups.
    return neutral_score + ((row_number % 100_000) + 1) * 1e-6


def write_asv19_fixture() -> None:
    source = select_prediction("asv19la_")
    observed_scores = score_map(source)
    rows = []
    with asv19_protocol_path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            fields = line.split()
            if len(fields) < 5:
                continue
            utterance_id = fields[1]
            label = 0 if fields[4] == "bonafide" else 1
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "label": label,
                    "bonafide_score": observed_scores.get(
                        utterance_id,
                        fallback_score(row_number),
                    ),
                }
            )
    destination = tested_results_dir / "asv19la_smoke_fixture.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"ASV19 fixture: {destination} ({len(rows)} rows; {len(observed_scores)} model rows)")


def write_asv21_fixture(track: str) -> None:
    prefix = f"asv21{track}_"
    source = select_prediction(prefix)
    observed_scores = score_map(source)
    names = (
        ["spk", "trial", "codec", "trans", "attack", "label", "trim", "subset"]
        if track == "la"
        else [
            "speaker",
            "trial",
            "compr",
            "source",
            "attack",
            "label",
            "trim",
            "subset",
            "vocoder",
            "task",
            "team",
            "gender-pair",
            "language",
        ]
    )
    metadata_path = asv21_keys_dir / track.upper() / "CM" / "trial_metadata.txt"
    metadata = pd.read_csv(
        metadata_path,
        sep=r"\s+",
        names=names,
        usecols=["trial"],
        engine="python",
    )
    destination = tested_results_dir / f"{prefix}smoke_fixture.csv"
    fixture = pd.DataFrame(
        {
            "utterance_id": metadata["trial"].astype(str),
            "bonafide_score": [
                observed_scores.get(
                    str(utterance_id),
                    fallback_score(row_number),
                )
                for row_number, utterance_id in enumerate(metadata["trial"])
            ],
        }
    )
    fixture.to_csv(destination, index=False)
    print(f"ASV21-{track.upper()} fixture: {destination} ({len(fixture)} rows; {len(observed_scores)} model rows)")


write_asv19_fixture()
for track in tracks:
    write_asv21_fixture(track)
PY
}

run_step() {
  local title="$1"
  shift
  echo
  echo "[$title]"
  printf '  '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != true ]]; then
    "$@"
  fi
}

echo "================================================================================"
echo "Optimized ADD smoke pipeline"
echo "Run directory:     $RUN_DIR"
echo "Output root:       $OUTPUT_ROOT_ABS"
echo "ASV21 datasets:    $ASV21_DATASETS"
echo "Checkpoint mode:   $CHECKPOINT_MODE"
echo "Max test samples:  $MAX_TEST_SAMPLES"
echo "================================================================================"

run_step "train" python "$SCRIPT_DIR/train.py" \
  --config "$COMMON_CONFIG" \
  --config "$TEMP_CONFIG"

run_step "test ASV19 LA" "$SCRIPT_DIR/test.sh" asv19 \
  --model_path "$RUN_DIR" \
  --checkpoint_mode "$CHECKPOINT_MODE" \
  --max_predict_samples "$MAX_TEST_SAMPLES" \
  --per_device_eval_batch_size 2 \
  --dataloader_num_workers 0

run_step "test ASV21" "$SCRIPT_DIR/test.sh" asv21 \
  --model_path "$RUN_DIR" \
  --datasets "$ASV21_DATASETS" \
  --checkpoint_mode "$CHECKPOINT_MODE" \
  --max_predict_samples "$MAX_TEST_SAMPLES" \
  --per_device_eval_batch_size 2 \
  --dataloader_num_workers 0

prepare_score_fixtures

run_step "score ASV19 LA" "$SCRIPT_DIR/score.sh" asv19 \
  --results-dir "$RUN_DIR" \
  --file-name "asv19la_smoke_fixture.csv"

ASV21_SCORE_ARGS=(
  "$SCRIPT_DIR/score.sh"
  asv21
  --results-dir "$RUN_DIR"
  --subset eval
)
if [[ "$ASV21_DATASETS" == "la" || "$ASV21_DATASETS" == "both" ]]; then
  ASV21_SCORE_ARGS+=(--file-name "asv21la_smoke_fixture.csv")
fi
if [[ "$ASV21_DATASETS" == "df" || "$ASV21_DATASETS" == "both" ]]; then
  ASV21_SCORE_ARGS+=(--file-name "asv21df_smoke_fixture.csv")
fi
run_step "score ASV21" "${ASV21_SCORE_ARGS[@]}"

echo
echo "Smoke pipeline complete. Results: $RUN_DIR/tested_results"

# ./code_optimize/smoke_run_pipeline.sh
# ./code_optimize/smoke_run_pipeline.sh --datasets both
# ./code_optimize/smoke_run_pipeline.sh --output-root outputs/smoke
