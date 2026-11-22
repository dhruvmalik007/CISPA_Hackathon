"""Submit a state dict to the challenge server, matching the official submission.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from unlearning.task import validate_submission

DEFAULT_SERVER_URL = "http://34.147.229.55"
DEFAULT_TASK_ID = "31-unlearning"


def submit_file(
    path: str | Path,
    api_key: str | None = None,
    server_url: str = DEFAULT_SERVER_URL,
    task_id: str = DEFAULT_TASK_ID,
    timeout: tuple[float, float] = (10.0, 600.0),
    validate: bool = True,
) -> dict[str, Any]:
    import requests

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in (".pt", ".pth"):
        raise ValueError(f"{task_id} expects a .pt or .pth file, got {path.name}")

    key = api_key or os.environ.get("HACKATHON_API_KEY", "")
    if not key.strip():
        raise ValueError("Set HACKATHON_API_KEY (or pass api_key) before submitting.")

    report = validate_submission(path) if validate else None
    if report is not None and not report["ok"]:
        raise ValueError(
            "submission failed the local contract check: "
            f"strict_load={report['strict_load']} float32={not report['non_float32']} "
            f"finite={not report['non_finite']} size_ok={report['within_size_limit']}"
        )

    with path.open("rb") as handle:
        response = requests.post(
            f"{server_url.rstrip('/')}/submit/{task_id}",
            headers={"X-API-Key": key},
            files={"file": (path.name, handle)},
            timeout=timeout,
        )
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return {
        "status_code": response.status_code,
        "score": body.get("score"),
        "status": body.get("status"),
        "message": body.get("message"),
        "previous_best": body.get("previous_best"),
        "improved": body.get("improved"),
        "body": body,
        "validation": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="path to a .pt/.pth state dict")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    try:
        result = submit_file(
            args.submission,
            api_key=args.api_key,
            server_url=args.server_url,
            task_id=args.task_id,
            validate=not args.skip_validation,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps({k: v for k, v in result.items() if k not in ("body", "validation")}, indent=2))
    if result["status_code"] != 200 or result["status"] != "success":
        print(json.dumps(result["body"], indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
