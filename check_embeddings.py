#!/usr/bin/env python3
"""
Scan embedding .db files and report runs with fewer than expected samples.

Usage:
    python check_embeddings.py --emb-dir /path/to/embeddings --expected 5000

Reports for each DB file the run_id counts and highlights run_ids with counts < expected.
"""
import os
import sqlite3
import argparse

DEFAULT_EMB_DIR = "/CECI/home/ucl/irec/darimez/proj"


def inspect_db(db_path, expected=5000):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    counts = {}
    # Try embeddings table first
    try:
        c.execute("SELECT run_id, COUNT(*) FROM embeddings GROUP BY run_id")
        for r in c.fetchall():
            counts[r[0]] = r[1]
    except Exception:
        # Fallback to sqlvector vec_projections
        try:
            c.execute("SELECT run_id, COUNT(*) FROM vec_projections GROUP BY run_id")
            for r in c.fetchall():
                counts[r[0]] = r[1]
        except Exception:
            # Nothing we can do
            conn.close()
            return None
    conn.close()
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", default=DEFAULT_EMB_DIR)
    p.add_argument("--expected", type=int, default=5000)
    p.add_argument("--show-all", action="store_true")
    p.add_argument("--clean", action="store_true", help="Delete runs with count < --expected")
    p.add_argument("--yes", action="store_true", help="Assume yes for deletions (non-interactive)")
    args = p.parse_args()

    if not os.path.exists(args.emb_dir):
        print(f"Embeddings directory not found: {args.emb_dir}")
        return

    dbs = [os.path.join(args.emb_dir, f) for f in os.listdir(args.emb_dir) if f.endswith('.db') and f != 'attention.db']
    if not dbs:
        print("No embedding .db files found")
        return

    total_flagged = 0
    for db in sorted(dbs):
        counts = inspect_db(db, expected=args.expected)
        if counts is None:
            print(f"{os.path.basename(db)}: could not read embeddings or vec_projections table")
            continue
        # summarize
        low = {rid: cnt for rid, cnt in counts.items() if cnt < args.expected}
        if low:
            print(f"{os.path.basename(db)}: {len(low)}/{len(counts)} runs below expected ({args.expected})")
            total_flagged += len(low)
            if args.show_all:
                for rid, cnt in sorted(low.items(), key=lambda x: x[1]):
                    print(f"  {rid} -> {cnt}")
            else:
                for rid, cnt in sorted(low.items(), key=lambda x: x[1])[:10]:
                    print(f"  {rid} -> {cnt}")
                if len(low) > 10:
                    print(f"  ... ({len(low)-10} more)")
            # Optionally remove low-count runs from this DB
            if args.clean:
                if not args.yes:
                    confirm = input(f"Delete {len(low)} runs from {db}? [y/N]: ")
                    do_delete = confirm.lower() == "y"
                else:
                    do_delete = True
                if do_delete:
                    conn = sqlite3.connect(db)
                    c = conn.cursor()
                    removed = 0
                    for rid in low.keys():
                        try:
                            # try both tables where projections/embeddings may live
                            c.execute("DELETE FROM embeddings WHERE run_id = ?", (rid,))
                        except Exception:
                            pass
                        try:
                            c.execute("DELETE FROM vec_projections WHERE run_id = ?", (rid,))
                        except Exception:
                            pass
                        try:
                            c.execute("DELETE FROM metadata WHERE run_id = ?", (rid,))
                        except Exception:
                            pass
                        removed += 1
                    try:
                        conn.commit()
                    except Exception:
                        pass
                    conn.close()
                    print(f"  Deleted {removed} runs from {os.path.basename(db)}")
        else:
            print(f"{os.path.basename(db)}: all runs >= {args.expected} (checked {len(counts)} runs)")
    print(f"\nTotal runs below expected: {total_flagged}")


if __name__ == '__main__':
    main()
