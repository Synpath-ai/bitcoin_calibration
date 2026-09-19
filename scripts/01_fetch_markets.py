"""Discover all resolved daily BTC Up/Down markets and cache raw + a clean index."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import market_matching as mm

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    print("Fetching resolved daily BTC Up/Down markets from Gamma series 'btc-up-or-down-daily'...")
    accepted, excluded = mm.discover_daily_markets(closed=True)
    print(f"Accepted candidates: {len(accepted)}")
    print(f"Excluded at discovery stage: {len(excluded)}")

    resolved_rows = []
    unresolved = []
    for c in accepted:
        outcome, reason = mm.resolve_outcome(c)
        row = asdict(c)
        if outcome is None:
            unresolved.append(mm.ExclusionRecord(c.slug, c.event_id, "unresolvable_outcome", reason))
            continue
        row["outcome_up"] = outcome
        resolved_rows.append(row)

    (PROCESSED / "markets_accepted.json").write_text(json.dumps(resolved_rows, indent=2))
    all_exclusions = [asdict(e) for e in excluded] + [asdict(e) for e in unresolved]
    (PROCESSED / "markets_excluded.json").write_text(json.dumps(all_exclusions, indent=2))

    print(f"Resolved markets with clean Up/Down outcome: {len(resolved_rows)}")
    print(f"Total excluded (discovery + outcome mapping): {len(all_exclusions)}")

    reasons = {}
    for e in all_exclusions:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    print("Exclusion reasons:", json.dumps(reasons, indent=2))


if __name__ == "__main__":
    main()
