#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Submit DS4 vLLM changes through the experiencenow fork.

This helper exists to keep the DS4 Spark deployment workflow out of the
upstream vllm-project remote. The Sparks pull experiencenow-ai/vllm, so DS4
performance fixes must be pushed and merged there before deployment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PUSH_REMOTE = "experiencenow"
DEFAULT_REPO = "experiencenow-ai/vllm"
DEFAULT_HEAD_OWNER = "experiencenow-ai"
DEFAULT_BASE = "main"


def run(args: list[str], *, capture: bool = False) -> str:
    proc = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if proc.returncode != 0:
        if capture and proc.stdout:
            sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    return proc.stdout.strip() if capture and proc.stdout else ""


def current_branch() -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True)


def remote_url(remote: str) -> str:
    return run(["git", "remote", "get-url", remote], capture=True)


def ensure_clean_tree() -> None:
    status = run(["git", "status", "--porcelain"], capture=True)
    if status != "":
        raise SystemExit(
            "working tree is dirty; commit or stash before submitting a DS4 PR"
        )


def find_existing_pr(repo: str, head: str) -> str | None:
    out = run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--head",
            head,
            "--json",
            "number",
            "--limit",
            "1",
        ],
        capture=True,
    )
    data = json.loads(out or "[]")
    if not data:
        return None
    return str(data[0]["number"])


def create_pr(repo: str, head: str, base: str, title: str, body: str) -> str:
    return run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--head",
            head,
            "--base",
            base,
            "--title",
            title,
            "--body",
            body,
        ],
        capture=True,
    )


def pr_number_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--push-remote", default=DEFAULT_PUSH_REMOTE)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--head-owner", default=DEFAULT_HEAD_OWNER)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--delete-branch", action="store_true")
    args = parser.parse_args()

    branch = current_branch()
    if branch == "" or branch == args.base:
        raise SystemExit(f"refusing to submit from branch {branch!r}")
    ensure_clean_tree()

    push_url = remote_url(args.push_remote)
    if "experiencenow-ai/vllm" not in push_url:
        raise SystemExit(
            f"refusing push remote {args.push_remote!r}: {push_url!r} "
            "does not point at experiencenow-ai/vllm"
        )
    origin = remote_url("origin")
    if "vllm-project/vllm" not in origin:
        print(
            f"warning: origin is not upstream vllm-project/vllm: {origin}",
            file=sys.stderr,
        )

    print(f"pushing {branch} to {args.push_remote} ({push_url})")
    run(["git", "push", "-u", args.push_remote, branch])

    head = f"{args.head_owner}:{branch}"
    pr_number = find_existing_pr(args.repo, branch)
    if pr_number is None:
        url = create_pr(args.repo, head, args.base, args.title, args.body)
        pr_number = pr_number_from_url(url)
        print(url)
    else:
        print(f"https://github.com/{args.repo}/pull/{pr_number}")

    if args.merge:
        merge_args = [
            "gh",
            "pr",
            "merge",
            "--repo",
            args.repo,
            pr_number,
            "--squash",
        ]
        if args.delete_branch:
            merge_args.append("--delete-branch")
        run(merge_args)
        merged = run(
            [
                "gh",
                "pr",
                "view",
                "--repo",
                args.repo,
                pr_number,
                "--json",
                "state,mergedAt,mergeCommit,title",
            ],
            capture=True,
        )
        print(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
