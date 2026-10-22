"""Download the UCI 'Default of Credit Card Clients' dataset (id 350, CC BY 4.0)."""
import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "default_of_credit_card_clients.xls"
    if target.exists():
        print(f"already downloaded: {target}")
        return
    print(f"downloading {URL}")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = [n for n in z.namelist() if n.endswith(".xls")][0]
        target.write_bytes(z.read(name))
    print(f"saved {target} ({target.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
