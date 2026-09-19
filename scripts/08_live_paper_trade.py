"""Run one live paper-trading check-in against the current active BTC market.

Usage: python scripts/08_live_paper_trade.py [daily|hourly]  (default: daily)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paper_trader as pt


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "daily"
    result = pt.run_live_paper_trade(family=family)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
