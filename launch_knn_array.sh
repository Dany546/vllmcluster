#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=knn
#SBATCH --time=03:00:00

#SBATCH --output=/auto/home/users/d/a/darimez/MIRO/vllmcluster/knn_%A_%a.out
#SBATCH --error=/auto/home/users/d/a/darimez/MIRO/vllmcluster/knn_%A_%a.err

module load releases/2023b
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.5.0

cd ~/MIRO/vllmcluster
source dino/bin/activate

# TABLES_FILE should contain one table name or path per line. Prefer basenames (no .db) or full .db paths.
TABLES_FILE=${TABLES_FILE:-/tmp/knn_tables_${USER}.txt}
# Optionally override where embeddings .db files live
DB_PATH=${DB_PATH:-/globalscratch/ucl/irec/darimez/dino/embeddings}

# If a tables file isn't provided, auto-generate it by enumerating DB_PATH
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
fi

# Determine index (supports both array task id and manual invocation)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=${TABLE_INDEX:-0}
else
    IDX=$SLURM_ARRAY_TASK_ID
fi

TABLE_LINE=$(sed -n "$((IDX+1))p" "$TABLES_FILE")
if [ -z "$TABLE_LINE" ]; then
    echo "No table found at index $IDX in $TABLES_FILE"
    exit 3
fi

echo "[${SLURM_JOB_ID:-local}] Processing table (index ${IDX}): $TABLE_LINE"

# Run KNN for this table. Use --debug for faster local testing; jobs on cluster should omit --debug.
python main.py --knn --table "$TABLE_LINE"

# Exit code propagated
exit $?
