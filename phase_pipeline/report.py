"""Write a phase report to disk."""

import json


def write_report(output_dir, name, report):
    """Write one report as JSON.

    Args:
        output_dir (Path): directory to write into, created if absent.
        name (str): filename stem, e.g. "phase1_2024".
        report (dict): the report.

    Returns:
        Path: the file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, default=str),
                    encoding="utf-8")
    return path
