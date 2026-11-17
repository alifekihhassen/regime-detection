"""Construction des features quotidiennes à partir du S&P 500 et du VIX."""

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Log-rendements journaliers : ln(P_t / P_{t-1})."""
    return np.log(prices).diff()


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Volatilité glissante annualisée."""
    return returns.rolling(window).std() * np.sqrt(252)


def momentum(prices: pd.Series, window: int = 63) -> pd.Series:
    """Rendement cumulé (en log) sur `window` jours."""
    return np.log(prices).diff(window)


def drawdown(prices: pd.Series) -> pd.Series:
    """Écart relatif au plus haut historique (valeurs <= 0)."""
    return prices / prices.cummax() - 1


def build_features(spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Assemble toutes les features dans un DataFrame sans NaN.

    Toutes les valeurs du jour t n'utilisent que des données disponibles à la clôture de t.
    """
    ret = log_returns(spx)
    features = pd.concat(
        {
            "ret": ret,
            "vol_21": realized_vol(ret),
            "mom_63": momentum(spx),
            "drawdown": drawdown(spx),
            "vix": vix,
        },
        axis=1,
    )
    return features.dropna()


def hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Les 3 variables d'entrée du HMM multivarié : ret, vol_21 et log(VIX).

    L'ordre des colonnes compte : la colonne 1 (vol_21) sert à ordonner les états.
    """
    return pd.DataFrame(
        {
            "ret": df["ret"],
            "vol_21": df["vol_21"],
            "log_vix": np.log(df["vix"]),
        }
    )
