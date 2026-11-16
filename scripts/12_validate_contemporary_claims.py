"""CLI validator for contemporary claims benchmark CSV (Brief #4 §8.4).

Ensures:
- Valid header: claim_id, claim, label, source_url, source_date, note, split
- Valid labels: real, fake, unverifiable
- No duplicate claim_id or claim texts
- All source_urls strictly belong to search whitelist include_domains
- Valid ISO dates (YYYY-MM-DD)
- Clean split separation: at least 10 tune rows, remaining test rows
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from vnfake.config import load_search_config
from vnfake.data.cleaning import normalize_text
from vnfake.retrieval.web_search import is_allowed_domain
from vnfake.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS = (
    "claim_id",
    "claim",
    "label",
    "source_url",
    "source_date",
    "note",
    "split",
)
_VALID_LABELS = frozenset({"real", "fake", "unverifiable"})
_VALID_SPLITS = frozenset({"tune", "test"})


def validate_contemporary_claims(
    csv_path: Path,
    include_domains: list[str],
    *,
    min_tune_rows: int = 10,
    allow_empty_rows: bool = False,
) -> tuple[bool, list[str]]:
    """Validate contemporary claims CSV against Brief #4 §8.4 rules."""
    errors: list[str] = []
    if not csv_path.exists():
        return False, [f"File not found: {csv_path}"]

    with csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return False, ["CSV file is empty or missing headers."]
        missing_cols = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            return False, [f"Missing required columns: {missing_cols}"]

        seen_ids: set[str] = set()
        seen_claims: set[str] = set()
        tune_count = 0
        test_count = 0
        row_idx = 0

        for row in reader:
            row_idx += 1
            cid = (row.get("claim_id") or "").strip()
            claim = normalize_text(row.get("claim") or "")
            label = (row.get("label") or "").strip().lower()
            url = (row.get("source_url") or "").strip()
            date_str = (row.get("source_date") or "").strip()
            split = (row.get("split") or "").strip().lower()

            if not cid and not claim:
                if allow_empty_rows:
                    continue
                errors.append(f"Row {row_idx}: Both claim_id and claim are empty.")
                continue

            if not cid:
                errors.append(f"Row {row_idx}: Missing claim_id.")
            elif cid in seen_ids:
                errors.append(f"Row {row_idx}: Duplicate claim_id {cid!r}.")
            else:
                seen_ids.add(cid)

            if not claim:
                if not allow_empty_rows:
                    errors.append(f"Row {row_idx} ({cid}): Missing claim text.")
            else:
                claim_norm = claim.lower()
                if claim_norm in seen_claims:
                    errors.append(f"Row {row_idx} ({cid}): Duplicate claim text.")
                else:
                    seen_claims.add(claim_norm)

            if label:
                if label not in _VALID_LABELS:
                    errors.append(
                        f"Row {row_idx} ({cid}): Invalid label {label!r}; must be one of {sorted(_VALID_LABELS)}."
                    )
            elif not allow_empty_rows:
                errors.append(f"Row {row_idx} ({cid}): Missing label.")

            if url:
                if not is_allowed_domain(url, include_domains):
                    errors.append(
                        f"Row {row_idx} ({cid}): source_url {url!r} violates include_domains whitelist."
                    )
            elif not allow_empty_rows:
                errors.append(f"Row {row_idx} ({cid}): Missing source_url.")

            if date_str:
                try:
                    datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    errors.append(
                        f"Row {row_idx} ({cid}): source_date {date_str!r} is not in YYYY-MM-DD format."
                    )

            if split not in _VALID_SPLITS:
                errors.append(
                    f"Row {row_idx} ({cid}): Invalid split {split!r}; must be 'tune' or 'test'."
                )
            elif split == "tune":
                tune_count += 1
            else:
                test_count += 1

    if not allow_empty_rows and tune_count < min_tune_rows:
        errors.append(
            f"Insufficient tune rows: found {tune_count}, required at least {min_tune_rows}."
        )

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate contemporary claims benchmark CSV (Brief #4 §8.4)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/contemporary/claims_template.csv"),
        help="Path to CSV file to validate",
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path("configs/search.yaml"),
        help="Search config path with include_domains",
    )
    parser.add_argument(
        "--allow-template-placeholders",
        action="store_true",
        help="Allow empty draft rows in template file",
    )
    args = parser.parse_args()

    search_cfg = load_search_config(args.search_config)
    ok, errors = validate_contemporary_claims(
        args.csv,
        search_cfg.include_domains,
        allow_empty_rows=args.allow_template_placeholders,
    )

    if ok:
        logger.info("Validation passed for %s", args.csv)
        print(f"OK: {args.csv} is fully valid.")
        return 0

    logger.error("Validation failed with %d error(s):", len(errors))
    for err in errors:
        print(f"  - [FAIL] {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
