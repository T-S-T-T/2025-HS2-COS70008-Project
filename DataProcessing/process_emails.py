import json
import re
from pathlib import Path
from datetime import datetime
from email import policy
from email.parser import BytesParser

import pandas as pd

# ——— CONFIG ——————————————————————————————————————————

BASE_DIR = Path(__file__).resolve().parent.parent
MAILDIR = BASE_DIR / "data" / "maildir"
OUTPUT_DIR = BASE_DIR / "data"
PROCESSED_CSV = OUTPUT_DIR / "processed_emails.csv"
EMAIL_INDEX_JSON = OUTPUT_DIR / "email_index.json"

# ——— UTILITIES ———————————————————————————————————————

def find_email_files(root: Path):
    """Recursively yield every file under `root` (regardless of suffix)."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path

def normalize_addresses(raw: str) -> str:
    """Split on commas/semicolons, strip whitespace, re-join by semicolon."""
    if not raw:
        return ""
    parts = re.split(r"[;,]", raw)
    parts = [p.strip() for p in parts if p.strip()]
    return ";".join(parts)

def parse_email_file(path: Path) -> dict:
    """Parse headers and body into a flat dict."""
    with open(path, "rb") as fp:
        msg = BytesParser(policy=policy.default).parse(fp)

    # Headers
    message_id = msg.get("Message-ID", "").strip()
    date_hdr   = msg.get("Date", "").strip()
    sender     = msg.get("From", "").strip()
    to_raw     = msg.get("To", "")
    cc_raw     = msg.get("Cc", "")
    bcc_raw    = msg.get("Bcc", "")
    subject    = msg.get("Subject", "").strip()

    # Date → ISO8601
    try:
        # remove trailing parenthetical timezone if present
        cleaned = re.sub(r"\s*\([^)]*\)$", "", date_hdr)
        dt = datetime.strptime(cleaned, "%a, %d %b %Y %H:%M:%S %z")
        date_iso = dt.isoformat()
    except Exception:
        date_iso = date_hdr

    # Body (concatenate all text/plain parts)
    if msg.is_multipart():
        parts = [p.get_content() for p in msg.walk()
                 if p.get_content_type() == "text/plain"]
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

# ——— MAIN ———————————————————————————————————————————

def main():

    print("Running...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    index_map = {}

    for eml_path in find_email_files(MAILDIR):
        try:
            rec = parse_email_file(eml_path)
            mid = rec["message_id"]
            if not mid:
                continue
            records.append(rec)
            # store path relative to data/
            rel = eml_path.relative_to(BASE_DIR / "data")
            index_map[mid] = str(rel)
        except Exception as e:
            print(f"[WARN] failed to parse {eml_path}: {e}")

    # Dump CSV
    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_CSV, index=False)

    # Dump JSON index
    with open(EMAIL_INDEX_JSON, "w", encoding="utf-8") as jf:
        json.dump(index_map, jf, indent=2)

    print(f"Processed {len(df)} emails → {PROCESSED_CSV}")
    print(f"Index written → {EMAIL_INDEX_JSON}")

    print("Fninshed")

if __name__ == "__main__":
    main()