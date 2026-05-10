#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ENTRY = {
    "id": "pedrosakuma-gpu-bruteforce",
    "repo": "https://github.com/pedrosakuma/rinha-backend-2026-gpu-bruteforce",
}


def load_entries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open() as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON array")

    return data


def add_entry(entries: list[dict[str, str]]) -> bool:
    for entry in entries:
        if entry.get("id") == ENTRY["id"]:
            if entry.get("repo") != ENTRY["repo"]:
                raise SystemExit(
                    f"entry {ENTRY['id']} already exists with a different repo: {entry.get('repo')}"
                )
            return False

    entries.append(ENTRY.copy())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add this backend to rinha-de-backend-2026 participants/pedrosakuma.json"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="participants/pedrosakuma.json",
        help="Path to participants/pedrosakuma.json in a checkout of zanfranceschi/rinha-de-backend-2026",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the entry JSON instead of modifying a participants file",
    )
    args = parser.parse_args()

    if args.print_only:
        print(json.dumps(ENTRY, indent=2, ensure_ascii=False))
        return

    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_entries(path)
    changed = add_entry(entries)

    with path.open("w") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False)
        file.write("\n")

    status = "added" if changed else "already-present"
    print(f"{status}: {ENTRY['id']} -> {path}")


if __name__ == "__main__":
    main()
