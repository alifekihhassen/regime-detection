"""Téléchargement des prix avec yfinance, avec un cache local en CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def download_prices(ticker: str, start: str = "2000-01-01", end: str | None = None,
                    refresh: bool = False) -> pd.Series:
    """Prix de clôture ajustés d'un ticker, sous forme de pd.Series.

    Le premier appel télécharge les données et les met en cache dans data/raw/.
    Les appels suivants relisent le cache, sauf si refresh=True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{ticker.replace('^', '')}.csv"

    if refresh or not cache.exists():
        import yfinance as yf

        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"Aucune donnée reçue pour {ticker} (connexion à Yahoo ?)")
        # Les versions récentes de yfinance renvoient des colonnes multi-niveaux
        prices = df["Close"].squeeze("columns").dropna()
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "Date"
        prices.name = ticker
        prices.to_csv(cache)

    prices = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze("columns")
    prices.name = ticker
    return prices.loc[start:end]
