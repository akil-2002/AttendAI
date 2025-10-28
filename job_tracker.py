#!/usr/bin/env python
"""Simple CLI tool for tracking job applications."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional


DATA_PATH = Path(__file__).with_name("job_applications.json")


@dataclass
class JobApplication:
    """Container for a single job application."""

    company: str
    position: str
    status: str = "Applied"
    notes: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def load_applications(path: Path = DATA_PATH) -> List[JobApplication]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [JobApplication(**item) for item in data]


def save_applications(applications: Iterable[JobApplication], path: Path = DATA_PATH) -> None:
    data = [app.to_dict() for app in applications]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def add_application(args: argparse.Namespace) -> None:
    applications = load_applications()
    applications.append(
        JobApplication(company=args.company, position=args.position, status=args.status, notes=args.notes or "")
    )
    save_applications(applications)
    print(f"Added application for {args.position} @ {args.company} (status: {args.status}).")


def list_applications(_: argparse.Namespace) -> None:
    applications = load_applications()
    if not applications:
        print("No job applications tracked yet. Use 'add' to create one.")
        return

    widths = {
        "company": max(len("Company"), *(len(app.company) for app in applications)),
        "position": max(len("Position"), *(len(app.position) for app in applications)),
        "status": max(len("Status"), *(len(app.status) for app in applications)),
    }

    header = f"{ 'Company'.ljust(widths['company']) }  { 'Position'.ljust(widths['position']) }  { 'Status'.ljust(widths['status']) }  Notes"
    print(header)
    print("-" * len(header))
    for app in applications:
        notes = app.notes.replace("\n", " ")
        print(
            f"{app.company.ljust(widths['company'])}  "
            f"{app.position.ljust(widths['position'])}  "
            f"{app.status.ljust(widths['status'])}  "
            f"{notes}"
        )


def update_application(args: argparse.Namespace) -> None:
    applications = load_applications()
    index = args.index
    if index < 1 or index > len(applications):
        raise SystemExit(f"Invalid index {index}. There are {len(applications)} applications tracked.")

    app = applications[index - 1]
    if args.company:
        app.company = args.company
    if args.position:
        app.position = args.position
    if args.status:
        app.status = args.status
    if args.notes is not None:
        app.notes = args.notes

    save_applications(applications)
    print(f"Updated application #{index}.")


def delete_application(args: argparse.Namespace) -> None:
    applications = load_applications()
    index = args.index
    if index < 1 or index > len(applications):
        raise SystemExit(f"Invalid index {index}. There are {len(applications)} applications tracked.")

    removed = applications.pop(index - 1)
    save_applications(applications)
    print(f"Deleted application for {removed.position} @ {removed.company}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track job applications in a local JSON file.",
        epilog="Run without subcommands to open the interactive prompt.",
    )
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Add a new job application")
    add_parser.add_argument("company", help="Company name")
    add_parser.add_argument("position", help="Role or job title")
    add_parser.add_argument("--status", default="Applied", help="Current status of the application")
    add_parser.add_argument("--notes", default="", help="Optional notes to store with the application")
    add_parser.set_defaults(func=add_application)

    list_parser = subparsers.add_parser("list", help="List all tracked job applications")
    list_parser.set_defaults(func=list_applications)

    update_parser = subparsers.add_parser("update", help="Update an existing application by its index from the list command")
    update_parser.add_argument("index", type=int, help="1-based index of the application to update")
    update_parser.add_argument("--company", help="New company name")
    update_parser.add_argument("--position", help="New position title")
    update_parser.add_argument("--status", help="New status")
    update_parser.add_argument("--notes", help="Replace notes (use empty string to clear)")
    update_parser.set_defaults(func=update_application)

    delete_parser = subparsers.add_parser("delete", help="Delete an application by its index from the list command")
    delete_parser.add_argument("index", type=int, help="1-based index of the application to delete")
    delete_parser.set_defaults(func=delete_application)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        if sys.stdin.isatty():
            interactive_prompt()
        else:
            print("No subcommand provided; showing saved applications. Use --help for commands.")
            list_applications(SimpleNamespace())
        return

    args.func(args)


def interactive_prompt() -> None:
    """Simple REPL that lets users manage applications without subcommands."""

    MENU = (
        "\nJob Application Tracker",
        "1) List applications",
        "2) Add application",
        "3) Update application",
        "4) Delete application",
        "5) Quit",
    )

    while True:
        print("\n".join(MENU))
        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            list_applications(SimpleNamespace())
        elif choice == "2":
            company = input("Company: ").strip()
            position = input("Position: ").strip()
            status = input("Status [Applied]: ").strip() or "Applied"
            notes = input("Notes (optional): ").strip()
            add_application(
                SimpleNamespace(
                    company=company,
                    position=position,
                    status=status,
                    notes=notes,
                )
            )
        elif choice == "3":
            try:
                index = int(input("Entry number to update: ").strip())
            except ValueError:
                print("Please enter a valid number.")
                continue
            company = input("New company (leave blank to keep current): ").strip() or None
            position = input("New position (leave blank to keep current): ").strip() or None
            status = input("New status (leave blank to keep current): ").strip() or None
            notes = input("New notes (leave blank to keep current, type '-' to clear): ").strip()
            if notes == "":
                notes_value = None
            elif notes == "-":
                notes_value = ""
            else:
                notes_value = notes
            update_application(
                SimpleNamespace(
                    index=index,
                    company=company,
                    position=position,
                    status=status,
                    notes=notes_value,
                )
            )
        elif choice == "4":
            try:
                index = int(input("Entry number to delete: ").strip())
            except ValueError:
                print("Please enter a valid number.")
                continue
            delete_application(SimpleNamespace(index=index))
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Choose an option between 1 and 5.")


if __name__ == "__main__":
    main()
