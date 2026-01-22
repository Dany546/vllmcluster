#!/bin/bash
# Helper to build tables file and submit a SLURM array where each task runs KNN on one table.
# Usage: ./submit_knn_array.sh [DB_PATH] [TABLES_FILE]
# Options:
#   --dry-run, -n   Print the tables and the sbatch command without submitting

# Parse flags and positional args (supports --dry-run anywhere)
DRY_RUN=0
POS1=""
POS2=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [DB_PATH] [TABLES_FILE] [-n|--dry-run]"
            exit 0
            ;;
        *)
            if [ -z "$POS1" ]; then
                POS1="$1"
            else
                POS2="$1"
            fi
            shift
            ;;
    esac
done

DB_PATH=${POS1:-/globalscratch/ucl/irec/darimez/dino/embeddings}
TABLES_FILE=${POS2:-/tmp/knn_tables_${USER}.txt}

echo "Using DB_PATH=${DB_PATH}"
echo "Using TABLES_FILE=${TABLES_FILE}"

# If TABLES_FILE doesn't exist, generate it from DB_PATH; otherwise use existing file
if [ ! -f "$TABLES_FILE" ]; then
    echo "Tables file $TABLES_FILE does not exist; generating from $DB_PATH"
    if [ ! -d "$DB_PATH" ]; then
        echo "Embeddings directory $DB_PATH does not exist. Exiting."
        exit 2
    fi
    # Find .db files and write basenames (no .db) to TABLES_FILE
    find "$DB_PATH" -maxdepth 1 -type f -name "*.db" -printf "%f\n" | sed 's/\.db$//' | sort -u > "$TABLES_FILE"
    N=$(wc -l < "$TABLES_FILE" | tr -d '[:space:]')
    if [ "$N" -eq 0 ]; then
        echo "No .db files found in $DB_PATH. Exiting."
        rm -f "$TABLES_FILE"
        exit 2
    fi
    echo "Wrote $N table(s) to $TABLES_FILE"
else
    N=$(wc -l < "$TABLES_FILE" | tr -d '[:space:]')
    echo "Found existing $TABLES_FILE with $N table(s)"
fi

ARRAY_RANGE="0-$((N-1))"
# include DB_PATH in exported envs so launch script can regenerate if needed
SBATCH_CMD=(sbatch --array=${ARRAY_RANGE} --export=TABLES_FILE="$TABLES_FILE",DB_PATH="$DB_PATH" launch_knn_array.sh)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "-- DRY RUN --"
    echo "First 20 tables (or fewer):"
    head -n 20 "$TABLES_FILE" | nl -ba -w3 -s": " || true
    echo
    echo "Total tables: $N"
    echo
    echo "SBATCH command that would be executed:"
    echo "${SBATCH_CMD[*]}"
    echo
    echo "You can run a local single-task test with: TABLE_INDEX=0 bash launch_knn_array.sh"
    exit 0
fi

# Submit the array
echo "Submitting SLURM array 0-$((N-1)) ..."
${SBATCH_CMD[*]}
