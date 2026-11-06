#!/usr/bin/env python3
"""Maintain accounts for a running deployment.

This runs inside the inference container because it operates on the live
accounts database and needs the service's dependencies:

    docker exec -it vqa-infer .venv/bin/python -m deploy.web.manage_account reset-password me@example.com

The Makefile wraps the same command in the ``account-*`` targets.
"""

import argparse
import getpass

from deploy.config import cfg
from src import __version__

from .accounts import AccountStore, normalize_email


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without executing the command."""
    parser = argparse.ArgumentParser(
        description="Manage local Deep-VQA accounts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python -m deploy.web.manage_account disable user@example.com\n  python -m deploy.web.manage_account reset-password user@example.com",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("email", type=normalize_email, help="Registered account email")
    for action in ("disable", "enable", "reset-password"):
        subparsers.add_parser(action, parents=[common_parser])
    parser.add_argument("--version", action="version", version=f"Deep-VQA {__version__}")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[str, str]:
    """Produce validated options from argv (or the process command line)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.action, args.email


def run(action: str, email: str) -> None:
    """Consume parsed options and execute the command."""
    if action not in {"disable", "enable", "reset-password"}:
        raise ValueError(f"Unknown account operation: {action}")
    store = AccountStore(
        cfg.resolve_path(cfg.endpoints.accounts_sqlite_path),
        cfg.deployment.sqlite.max_database_bytes,
        cfg.deployment.sqlite.busy_timeout_seconds,
    )
    store.initialize()
    if action == "reset-password":
        first = getpass.getpass("New password: ")
        second = getpass.getpass("Repeat password: ")
        if first != second:
            raise ValueError("Passwords do not match")
        store.reset_password(email, first)
        print(f"Password reset and sessions revoked for {email}")
    else:
        store.set_disabled(email, action == "disable")
        print(f"Account {email}: {action}d")


def main(argv: list[str] | None = None) -> None:
    """Parse command-line options and execute the command."""
    action, email = parse_args(argv)
    try:
        run(action, email)
    except ValueError as exc:
        build_parser().error(str(exc))


if __name__ == "__main__":
    main()
