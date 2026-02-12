#!/usr/bin/env python3
"""
Clean validated_parameters.csv by:
1. Removing columns 2 & 3 (experiment, run_name)
2. Renaming column 1 from 'folder_name' to 'run_name'
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(SCRIPT_DIR, "validated_parameters.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "validated_parameters_cleaned.csv")


def clean_csv():
    """Remove experiment and run_name columns, rename folder_name to run_name."""
    rows = []

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        original_cols = reader.fieldnames

        print(f"Original columns: {original_cols}")

        # Keep all columns except 'experiment' and 'run_name'
        # Rename 'folder_name' to 'run_name'
        new_cols = []
        for col in original_cols:
            if col == "folder_name":
                new_cols.append("run_name")
            elif col in ["experiment", "run_name"]:
                continue  # Skip these columns
            else:
                new_cols.append(col)

        print(f"New columns: {new_cols}")

        for row in reader:
            new_row = {}
            for old_col, new_col in zip(original_cols,
                                        ["run_name" if c == "folder_name" else c for c in original_cols]):
                if old_col not in ["experiment", "run_name"]:
                    if old_col == "folder_name":
                        new_row["run_name"] = row[old_col]
                    else:
                        new_row[old_col] = row[old_col]
            rows.append(new_row)

    # Write cleaned CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Columns: {', '.join(new_cols)}")


if __name__ == "__main__":
    clean_csv()
