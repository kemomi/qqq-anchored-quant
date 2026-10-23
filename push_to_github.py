"""
One-Click GitHub Repository Publisher (Zero-Dependency HTTPS REST API)
======================================================================
Author: kemomi
License: MIT

Automatically creates `https://github.com/kemomi/qqq-anchored-quant` (if it does not
already exist) and uploads all project files (`qqq_quant_strategy.py`,
`live_trader_execution.py`, `test_qqq_strategy.py`, `README.md`,
`qqq_strategy_tearsheet.png`, `requirements.txt`, `LICENSE`, and `.github/workflows/ci.yml`)
directly via the official GitHub REST API v3 using only Python's standard library.

Usage on your local machine:
----------------------------
    export GITHUB_TOKEN="your_new_github_personal_access_token"
    python3 push_to_github.py --owner kemomi --repo qqq-anchored-quant
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error


MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 kemomi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

REQUIREMENTS_TXT = """numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
yfinance>=0.2.40
"""

GITHUB_ACTIONS_CI_YML = """name: QQQ Quant Strategy CI & Backtest Verification

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  verify-strategy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas matplotlib

      - name: Run Unit & Zero-Lookahead Bias Tests
        run: |
          python3 test_qqq_strategy.py

      - name: Run Multi-Mode Backtest & Dry-Run Rebalance Check
        run: |
          python3 qqq_quant_strategy.py --mode synthetic --strategy-type all --output-json report.json
          python3 live_trader_execution.py --broker dry_run --strategy-mode leveraged_etf
"""


def github_api_request(
    url: str,
    token: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "qqq-anchored-quant-uploader",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"message": err_body}


def ensure_repo_exists(owner: str, repo: str, token: str, private: bool = False) -> None:
    status, _ = github_api_request(
        f"https://api.github.com/repos/{owner}/{repo}", token=token, method="GET"
    )
    if status == 200:
        print(f"[*] Repository https://github.com/{owner}/{repo} already exists.")
        return

    print(f"[*] Creating new repository: https://github.com/{owner}/{repo} ...")
    create_payload = {
        "name": repo,
        "description": "QQQ-Anchored Multi-Mode Quantitative Trading Strategy, Backtesting Engine & Live Broker Execution Suite",
        "private": private,
        "auto_init": True,
    }
    c_status, c_data = github_api_request(
        "https://api.github.com/user/repos", token=token, method="POST", payload=create_payload
    )
    if c_status not in (200, 201):
        raise RuntimeError(f"Failed to create repository ({c_status}): {c_data}")
    print(f"[+] Created repository: {c_data.get('html_url')}")


def upload_file_to_github(
    owner: str,
    repo: str,
    repo_path: str,
    content_bytes: bytes,
    token: str,
    commit_message: str,
) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
    status, existing = github_api_request(url, token=token, method="GET")

    payload: Dict[str, Any] = {
        "message": commit_message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
    }
    if status == 200 and "sha" in existing:
        payload["sha"] = existing["sha"]

    put_status, put_data = github_api_request(url, token=token, method="PUT", payload=payload)
    if put_status not in (200, 201):
        raise RuntimeError(f"Failed to upload {repo_path} ({put_status}): {put_data}")
    print(f"[+] Uploaded {repo_path} ({len(content_bytes):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-Click GitHub Repository Publisher for QQQ-Anchored Quant Strategy"
    )
    parser.add_argument("--owner", type=str, default="kemomi", help="GitHub username or org.")
    parser.add_argument(
        "--repo", type=str, default="qqq-anchored-quant", help="Target repository name."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var).",
    )
    parser.add_argument("--private", action="store_true", help="Create private repository.")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "Error: Please provide your GitHub Token via `--token <TOKEN>` or `export GITHUB_TOKEN=<TOKEN>`."
        )

    ensure_repo_exists(args.owner, args.repo, args.token, private=args.private)

    base_dir = Path(__file__).resolve().parent
    local_files = [
        "qqq_quant_strategy.py",
        "qqq_options_overlay.py",
        "live_trader_execution.py",
        "test_qqq_strategy.py",
        "push_to_github.py",
        "qqq_quant_dashboard.html",
        "README.md",
        "qqq_strategy_tearsheet.png",
    ]

    for fname in local_files:
        fpath = base_dir / fname
        if fpath.exists():
            upload_file_to_github(
                args.owner,
                args.repo,
                fname,
                fpath.read_bytes(),
                args.token,
                f"feat: add/update {fname}",
            )

    upload_file_to_github(
        args.owner,
        args.repo,
        "LICENSE",
        MIT_LICENSE_TEXT.encode("utf-8"),
        args.token,
        "chore: add MIT LICENSE",
    )
    upload_file_to_github(
        args.owner,
        args.repo,
        "requirements.txt",
        REQUIREMENTS_TXT.encode("utf-8"),
        args.token,
        "chore: add requirements.txt",
    )
    upload_file_to_github(
        args.owner,
        args.repo,
        ".github/workflows/ci.yml",
        GITHUB_ACTIONS_CI_YML.encode("utf-8"),
        args.token,
        "ci: add GitHub Actions automated test & backtest workflow",
    )

    print("=" * 72)
    print(f" SUCCESS! Full open-source project published to:")
    print(f" https://github.com/{args.owner}/{args.repo}")
    print("=" * 72)


if __name__ == "__main__":
    main()
