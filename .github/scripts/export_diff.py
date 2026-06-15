import os
import difflib
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Baseline summary CSV (results_branch version)")
    parser.add_argument("--target", required=True, help="New summary CSV (PR branch version)")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--suffix", default="", help="Suffix for output filenames, e.g. '_detailed'")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    header = open(args.source).readlines()[0] if os.path.exists(args.source) else []

    source_csv = open(args.source).readlines() if os.path.exists(args.source) else [header]
    target_csv = open(args.target).readlines() if os.path.exists(args.target) else [header]

    added_lines = header
    removed_lines = header

    for line in difflib.unified_diff(source_csv, target_csv, fromfile="before", tofile="after"):
        if line.startswith("+") and not line.startswith("+++"):
            added_lines += line[1:]
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines += line[1:]

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f"added{args.suffix}_rows.csv"), "w") as f:
        f.write(added_lines)
    with open(os.path.join(args.output, f"removed{args.suffix}_rows.csv"), "w") as f:
        f.write(removed_lines)
