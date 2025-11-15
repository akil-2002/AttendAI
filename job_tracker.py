#!/usr/bin/env python
"""Simple CLI tool for tracking job applications."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Set, Tuple


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

    play_parser = subparsers.add_parser(
        "play",
        help="Chase your job applications in a miniature Pac-Man-style maze",
    )
    play_parser.set_defaults(func=play_game)

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


def play_game(_: argparse.Namespace) -> None:
    """Launch a lightweight maze mini-game for reviewing applications."""

    applications = load_applications()
    if not applications:
        print("No job applications to chase. Add some first!")
        return

    try:
        import curses
    except Exception as exc:  # pragma: no cover - environment dependent
        print("The play mode requires the 'curses' module, which is unavailable:", exc)
        return

    maze_template = """
############################
#............##............#
#.######.####.##.####.######
#.#    #.#  ....  #.#    #.#
#.#.##.#.#.######.#.#.##.#.#
#....#....#......#....#....#
###.######.####.######.###.#
#............##............#
#.######.####.##.####.######
#......#............#......#
############################
""".strip("\n")

    maze_rows = maze_template.splitlines()
    board_height = len(maze_rows)
    board_width = len(maze_rows[0])

    walls: Set[Tuple[int, int]] = set()
    open_tiles: Set[Tuple[int, int]] = set()
    pellet_candidates: List[Tuple[int, int]] = []
    for y, row in enumerate(maze_rows):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((y, x))
            else:
                open_tiles.add((y, x))
                if char == ".":
                    pellet_candidates.append((y, x))

    non_pellet_tiles = [pos for pos in sorted(open_tiles) if pos not in pellet_candidates]
    if non_pellet_tiles:
        start = non_pellet_tiles[0]
    else:
        start = (1, 1)
        if start not in open_tiles:
            start = sorted(open_tiles)[0]

    if len(pellet_candidates) < len(applications):
        extras_needed = len(applications) - len(pellet_candidates)
        additional = [pos for pos in sorted(open_tiles) if pos not in pellet_candidates and pos != start]
        pellet_candidates.extend(additional[:extras_needed])

    if not pellet_candidates:
        print("The maze layout is unavailable right now. Try again later.")
        return

    displayed_apps = applications[: len(pellet_candidates)]
    if len(displayed_apps) < len(applications):
        print(
            f"Only showing the first {len(displayed_apps)} applications out of {len(applications)} in play mode."
        )

    active_pellets = pellet_candidates[: len(displayed_apps)]
    pellet_lookup = {pos: app for pos, app in zip(active_pellets, displayed_apps)}
    pellets: Set[Tuple[int, int]] = set(active_pellets)

    def draw_game(stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        info_offset = board_height + 1
        required_height = board_height + 6
        required_width = board_width + 2

        max_y, max_x = stdscr.getmaxyx()
        if max_y < required_height or max_x < required_width:
            stdscr.clear()
            warning = (
                "Terminal too small for the maze. Resize to at least "
                f"{required_width}x{required_height} and try again."
            )
            stdscr.addnstr(0, 0, warning, max_x - 1)
            stdscr.addnstr(2, 0, "Press any key to return to the shell.", max_x - 1)
            stdscr.refresh()
            stdscr.getch()
            return

        player_y, player_x = start
        collected: Set[Tuple[int, int]] = set()
        last_details: List[str] = []
        instructions = "Use arrow keys or WASD to move. Press Q to quit."

        while True:
            stdscr.erase()

            for y in range(board_height):
                for x in range(board_width):
                    if (y, x) in walls:
                        char = "#"
                    elif (y, x) == (player_y, player_x):
                        char = "C"
                    elif (y, x) in pellets and (y, x) not in collected:
                        char = "."
                    else:
                        char = " "
                    try:
                        stdscr.addch(y, x, char)
                    except curses.error:
                        pass

            progress_line = f"Applications captured: {len(collected)}/{len(pellets)}"
            lines_to_show = [progress_line, instructions]
            lines_to_show.extend(last_details)

            if len(collected) == len(pellets):
                lines_to_show.append("All applications collected! Press Q to exit.")

            for offset, line in enumerate(lines_to_show):
                try:
                    stdscr.addnstr(info_offset + offset, 0, line.ljust(max_x - 1), max_x - 1)
                except curses.error:
                    pass

            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            direction = {
                curses.KEY_UP: (-1, 0),
                curses.KEY_DOWN: (1, 0),
                curses.KEY_LEFT: (0, -1),
                curses.KEY_RIGHT: (0, 1),
                ord("w"): (-1, 0),
                ord("W"): (-1, 0),
                ord("s"): (1, 0),
                ord("S"): (1, 0),
                ord("a"): (0, -1),
                ord("A"): (0, -1),
                ord("d"): (0, 1),
                ord("D"): (0, 1),
            }.get(key)

            if direction:
                dy, dx = direction
                new_y, new_x = player_y + dy, player_x + dx
                if (new_y, new_x) in open_tiles:
                    player_y, player_x = new_y, new_x

                    pos = (player_y, player_x)
                    if pos in pellets and pos not in collected:
                        collected.add(pos)
                        job = pellet_lookup[pos]
                        info_width = max(20, min(max_x - 1, 120))
                        summary_text = f"Captured: {job.position} @ {job.company} ({job.status})"
                        if len(summary_text) > info_width:
                            summary = textwrap.shorten(summary_text, width=info_width, placeholder="…")
                        else:
                            summary = summary_text
                        notes = job.notes.strip().replace("\n", " ")
                        notes = notes if notes else "No notes yet."
                        note_line = f"Notes: {notes}"
                        wrapped_notes = textwrap.wrap(note_line, width=info_width) or [note_line[:info_width]]
                        last_details = [summary, *wrapped_notes]

            if len(collected) == len(pellets):
                stdscr.nodelay(False)

        stdscr.nodelay(False)

    try:
        curses.wrapper(draw_game)
    except curses.error as exc:  # pragma: no cover - environment dependent
        print("Unable to start play mode due to terminal issue:", exc)


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
