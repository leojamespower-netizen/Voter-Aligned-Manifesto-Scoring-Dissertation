"""Write the sixteen survey records to data/voter_profiles/.

    python -m phase_pipeline.build_election_data --bes DIR --ipsos DIR

Derived files. If a reader changes, rerun; --check reports what differs.
"""

import argparse
import json
import sys
from pathlib import Path

from .bes_extract import build_election_data
from .ipsos_extract import prepare_ipsos
from .vote_shares import ELECTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "voter_profiles"


# every record, keyed {election}_{source}. Writes nothing
def generate(bes_dir, ipsos_dir):
    records = {}
    for election in ELECTIONS:
        for source in ("bes", "ipsos"):
            if source == "bes":
                text, meta = build_election_data(election, bes_dir)
            else:
                text, meta = prepare_ipsos(election, ipsos_dir)
            records[f"{election}_{source}"] = {"text": text, "metadata": meta}
    return records


# metadata goes to a separate file: it names the election
def write(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, record in sorted(records.items()):
        path = output_dir / f"{key}.txt"
        path.write_text(record["text"], encoding="utf-8")
        written.append(path)

    index = {key: record["metadata"] for key, record in sorted(records.items())}
    (output_dir / "metadata.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")
    return written


# which files on disk differ from what the readers now produce
def check(records, output_dir):
    stale = []
    for key, record in sorted(records.items()):
        path = output_dir / f"{key}.txt"
        if not path.is_file():
            stale.append(f"{key}: missing")
        elif path.read_text(encoding="utf-8") != record["text"]:
            stale.append(f"{key}: differs")
    return stale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bes", type=Path, required=True,
                        help="directory holding the BES .dta files")
    parser.add_argument("--ipsos", type=Path, required=True,
                        help="directory holding the Ipsos .txt files")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="report which files are out of date, write none")
    return parser.parse_args()


def main():
    args = parse_args()
    for directory in (args.bes, args.ipsos):
        if not directory.is_dir():
            sys.exit(f"Not a directory: {directory}")

    records = generate(args.bes, args.ipsos)

    if args.check:
        stale = check(records, args.output)
        if stale:
            print(f"{len(stale)} of {len(records)} out of date:")
            for line in stale:
                print(f"  {line}")
            sys.exit(1)
        print(f"All {len(records)} files match the readers.")
        return

    written = write(records, args.output)
    for path in written:
        chars = len(path.read_text(encoding="utf-8"))
        print(f"  {path.name:16} {chars:6,} chars")
    print(f"\nWrote {len(written)} records + metadata.json to {args.output}")


if __name__ == "__main__":
    main()
