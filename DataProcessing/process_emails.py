import json
import csv
import re
from pathlib import Path
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime

# ——— CONFIG ————————————————————————————————

BASE_DIR    = Path(__file__).resolve().parent.parent
MAILDIR     = BASE_DIR / "data" / "maildir"
OUTPUT_DIR  = BASE_DIR / "data" / "DataProcessing"   # ← changed

INDEX_FILE  = OUTPUT_DIR / "email_index.ndjson"
CSV_HEADERS = [
    "message_id", "date", "sender",
    "recipients", "cc", "bcc",
    "subject", "body"
]

# ——— HELPERS ———————————————————————————————

def find_email_files(root: Path):
    """Yield every file under root (Maildir emails often have no extension)."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path

def normalize_addresses(raw: str) -> str:
    """Split on commas/semicolons, strip, rejoin by semicolon."""
    if not raw:
        return ""
    parts = re.split(r"[;,]", raw)
    return ";".join(p.strip() for p in parts if p.strip())

def parse_email_file(path: Path) -> dict:
    """Parse headers and body; format date to ISO8601 if possible."""
    with open(path, "rb") as fp:
        msg = BytesParser(policy=policy.default).parse(fp)

    message_id = msg.get("Message-ID", "").strip()
    date_hdr   = msg.get("Date", "").strip()
    sender     = msg.get("From", "").strip()
    to_raw     = msg.get("To", "")
    cc_raw     = msg.get("Cc", "")
    bcc_raw    = msg.get("Bcc", "")
    subject    = msg.get("Subject", "").strip()

    # ISO8601 date
    try:
        dt = parsedate_to_datetime(date_hdr)
        date_iso = dt.isoformat()
    except Exception:
        date_iso = date_hdr

    # Body text
    if msg.is_multipart():
        parts = [
            part.get_content()
            for part in msg.walk()
            if part.get_content_type() == "text/plain"
        ]
        body = "\n".join(parts).strip()
    else:
        body = msg.get_content().strip()

    return {
        "message_id":  message_id,
        "date":        date_iso,
        "sender":      sender,
        "recipients":  normalize_addresses(to_raw),
        "cc":          normalize_addresses(cc_raw),
        "bcc":         normalize_addresses(bcc_raw),
        "subject":     subject,
        "body":        body,
    }

# ——— MAIN ————————————————————————————————

def main():
    # ensure output folder exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INDEX_FILE, "w", encoding="utf-8") as idx_fh:
        seen_months = set()

        for eml_path in find_email_files(MAILDIR):
            try:
                rec = parse_email_file(eml_path)
                mid = rec["message_id"]
                if not mid:
                    continue

                # partition by YYYY_MM
                year_month = rec["date"][:7].replace("-", "_") if rec["date"] else "unknown"
                csv_path   = OUTPUT_DIR / f"processed_emails_{year_month}.csv"
                write_hdr  = year_month not in seen_months

                # append to monthly CSV
                with open(csv_path, "a", newline="", encoding="utf-8") as csv_fh:
                    writer = csv.DictWriter(csv_fh, fieldnames=CSV_HEADERS)
                    if write_hdr:
                        writer.writeheader()
                        seen_months.add(year_month)
                    writer.writerow(rec)

                # append to NDJSON index
                rel = eml_path.relative_to(BASE_DIR / "data")
                idx_fh.write(json.dumps({"message_id": mid, "path": str(rel)}) + "\n")

            except Exception as e:
                print(f"[WARN] failed to parse {eml_path}: {e}")

    print(f"Index → {INDEX_FILE}")
    print("Created month files:", ", ".join(sorted(seen_months)))

if __name__ == "__main__":
    main()