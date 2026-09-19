"""Discover the most recent ~2,000 resolved HOURLY BTC Up/Down markets -- an
extended, higher-frequency sample used to test whether the short-horizon edge
found in the daily backtest (~103 trades) replicates with a much larger N.
See src/market_matching_hourly.py for why this is a separate module from the
daily family (different slug formats, different resolution-time source)."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import market_matching_hourly as mmh

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

MAX_EVENTS = 2000  # ~offset-pagination cap on /events; most recent N, newest first


def main():
    print(f"Fetching the most recent {MAX_EVENTS} resolved hourly BTC Up/Down markets...")
    accepted, excluded = mmh.discover_hourly_markets(max_events=MAX_EVENTS, ascending=False, closed=True)
    print(f"Accepted candidates: {len(accepted)}")
    print(f"Excluded at discovery stage: {len(excluded)}")

    resolved_rows, unresolved = [], []
    for c in accepted:
        outcome, reason = mmh.resolve_outcome(c)
        row = asdict(c)
        if outcome is None:
            unresolved.append(mmh.ExclusionRecord(c.slug, c.event_id, "unresolvable_outcome", reason))
            continue
        row["outcome_up"] = outcome
        resolved_rows.append(row)

    (PROCESSED / "hourly_markets_accepted.json").write_text(json.dumps(resolved_rows, indent=2))
    all_exclusions = [asdict(e) for e in excluded] + [asdict(e) for e in unresolved]
    (PROCESSED / "hourly_markets_excluded.json").write_text(json.dumps(all_exclusions, indent=2))

    print(f"Resolved markets with clean Up/Down outcome: {len(resolved_rows)}")
    print(f"Total excluded: {len(all_exclusions)}")
    reasons = {}
    for e in all_exclusions:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    print("Exclusion reasons:", json.dumps(reasons, indent=2))
    if resolved_rows:
        print(f"Date range: {min(r['start_date'] for r in resolved_rows)} .. "
              f"{max(r['end_date'] for r in resolved_rows)}")


if __name__ == "__main__":
    main()
