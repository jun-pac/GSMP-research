import argparse
import gzip
import os
from collections import Counter
from datetime import datetime
from typing import Iterable, Optional, Tuple


REGISTRATION_COL = 6


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_registration(value: str) -> Optional[datetime]:
    value = value.strip()
    if not value or value == "null":
        return None
    formats = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def iter_registered_users(path: str) -> Iterable[Tuple[int, datetime]]:
    with open_text(path) as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= REGISTRATION_COL:
                continue
            try:
                user_id = int(parts[0])
            except ValueError:
                user_id = line_num
            registered_at = parse_registration(parts[REGISTRATION_COL])
            if registered_at is not None:
                yield user_id, registered_at


def split_name(rank: int, total: int, train_prop: float, val_prop: float) -> str:
    train_end = int(total * train_prop)
    val_end = train_end + int(total * val_prop)
    if rank < train_end:
        return "train"
    if rank < val_end:
        return "val"
    return "test"


def year_split_name(registered_at: datetime, train_until: int, val_year: int, test_from: int) -> Optional[str]:
    year = registered_at.year
    if year <= train_until:
        return "train"
    if year == val_year:
        return "val"
    if year >= test_from:
        return "test"
    return None


def print_counter(title: str, counter: Counter, total: int, limit: Optional[int] = None) -> None:
    print(title)
    items = sorted(counter.items())
    if limit is not None:
        items = items[:limit]
    for key, count in items:
        print(f"  {key}: {count} ({count / total:.4%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report Pokec registration-time distribution and temporal splits."
    )
    parser.add_argument(
        "--profiles",
        required=True,
        help="Path to soc-pokec-profiles.txt or soc-pokec-profiles.txt.gz.",
    )
    parser.add_argument("--train-prop", type=float, default=0.5)
    parser.add_argument("--val-prop", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=("quantile", "year"), default="quantile")
    parser.add_argument("--train-until-year", type=int, default=2010)
    parser.add_argument("--val-year", type=int, default=2011)
    parser.add_argument("--test-from-year", type=int, default=2012)
    parser.add_argument("--write-splits", default=None,
                        help="Optional output TSV: user_id, registered_at, split.")
    args = parser.parse_args()

    if not os.path.exists(args.profiles):
        raise FileNotFoundError(args.profiles)

    records = sorted(iter_registered_users(args.profiles), key=lambda item: (item[1], item[0]))
    total = len(records)
    if total == 0:
        raise ValueError("No parseable registration timestamps found.")

    split_counts = Counter()
    split_year_counts = Counter()
    split_month_counts = Counter()
    year_counts = Counter()
    month_counts = Counter()

    output_rows = []
    split_ranges = {"train": [], "val": [], "test": []}
    for rank, (user_id, registered_at) in enumerate(records):
        if args.split_mode == "quantile":
            split = split_name(rank, total, args.train_prop, args.val_prop)
        else:
            split = year_split_name(
                registered_at,
                train_until=args.train_until_year,
                val_year=args.val_year,
                test_from=args.test_from_year,
            )
            if split is None:
                continue
        year = registered_at.strftime("%Y")
        month = registered_at.strftime("%Y-%m")
        split_counts[split] += 1
        split_ranges[split].append(registered_at)
        year_counts[year] += 1
        month_counts[month] += 1
        split_year_counts[(split, year)] += 1
        split_month_counts[(split, month)] += 1
        if args.write_splits:
            output_rows.append((user_id, registered_at.isoformat(sep=" "), split))

    split_total = sum(split_counts.values())
    print(f"registered users: {total}")
    print(f"users included in split: {split_total}")
    print(f"first registration: {records[0][1]}")
    print(f"last registration: {records[-1][1]}")
    print_counter("split counts", split_counts, split_total)
    print_counter("year distribution", year_counts, total)

    print("split by year")
    for split in ("train", "val", "test"):
        split_total = split_counts[split]
        print(f"  {split}:")
        for (split_key, year), count in sorted(split_year_counts.items()):
            if split_key == split:
                print(f"    {year}: {count} ({count / split_total:.4%})")

    print("split date ranges")
    for split in ("train", "val", "test"):
        first = min(split_ranges[split])
        last = max(split_ranges[split])
        print(f"  {split}: {first} to {last}")

    if args.write_splits:
        with open(args.write_splits, "w", encoding="utf-8") as f:
            f.write("user_id\tregistered_at\tsplit\n")
            for user_id, registered_at, split in output_rows:
                f.write(f"{user_id}\t{registered_at}\t{split}\n")
        print(f"wrote {args.write_splits}")


if __name__ == "__main__":
    main()
