"""Tidy the raw UCI spreadsheet into data/processed/credit_clean.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"
OUT = ROOT / "data" / "processed" / "credit_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "DEFAULT"})
    df = df.drop(columns=["ID"])
    # Undocumented codes: EDUCATION 0/5/6 -> 4 ("others"), MARRIAGE 0 -> 3 ("others")
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].between(1, 4), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].between(1, 3), 3)
    # Repayment status: -2/-1/0 all mean "no delay" in practice
    return df.astype(int)


def main() -> None:
    df = pd.read_excel(RAW, header=1)
    print(f"raw rows: {len(df):,}, columns: {df.shape[1]}")
    df = clean(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"default rate: {df['DEFAULT'].mean():.2%}; saved {OUT}")


if __name__ == "__main__":
    main()
