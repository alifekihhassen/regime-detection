"""Backtest de stratégies d'exposition et métriques de performance."""

import numpy as np
import pandas as pd

GRADED_WEIGHTS = {"calme": 1.0, "tendu": 1.0, "nervosite": 0.7, "crise": 0.3}


def backtest(expo: pd.Series, r: pd.Series, cost: float = 0.0005) -> pd.Series:
    """Rendements quotidiens d'une stratégie.

    expo : exposition décidée à la clôture du jour t, appliquée au rendement de t+1.
    r : rendements SIMPLES de l'indice (un portefeuille investi à 30 % gagne 30 %
        du rendement simple, pas du log-rendement).
    cost : coût proportionnel à chaque variation d'exposition (5 pb par défaut).
    """
    expo_applied = expo.shift(1)                     # décider à t, investir à t+1
    costs = cost * expo_applied.diff().abs()         # coût à chaque changement
    rets = expo_applied * r - costs
    return rets.dropna()


def metrics(rets: pd.Series, expo: pd.Series) -> dict:
    """Rendement, volatilité, Sharpe, max drawdown, Calmar, exposition et rotation."""
    value = (1 + rets).cumprod()
    years = len(rets) / 252

    ann_return = value.iloc[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    max_dd = (value / value.cummax() - 1).min()

    return {
        "Rendement annualisé": ann_return,
        "Volatilité": vol,
        "Sharpe": ann_return / vol,
        "Max drawdown": max_dd,
        "Calmar": ann_return / abs(max_dd),
        "Exposition moyenne": expo.mean(),
        "Rotation annuelle": expo.diff().abs().sum() / years,
    }


METRIC_FORMATS = {
    "Rendement annualisé": "{:.1%}", "Volatilité": "{:.1%}", "Sharpe": "{:.2f}",
    "Max drawdown": "{:.1%}", "Calmar": "{:.2f}",
    "Exposition moyenne": "{:.0%}", "Rotation annuelle": "{:.1f}",
}


def regime_exposures(probs: pd.DataFrame, crisis_expo: float = 0.3,
                     weights: dict = GRADED_WEIGHTS) -> dict:
    """Les trois règles d'exposition basées sur les régimes."""
    idx = probs.index
    return {
        # crisis_expo si le régime le plus probable est la crise, sinon 100 %
        "Binaire": pd.Series(
            np.where(probs.idxmax(axis=1) == "crise", crisis_expo, 1.0), index=idx
        ),
        # Réduction proportionnelle à la probabilité de crise
        "Proportionnelle": 1 - (1 - crisis_expo) * probs["crise"],
        # Exposition par régime, pondérée par les probabilités des 4 régimes
        "Graduée": probs[list(weights)] @ pd.Series(weights),
    }


def vol_target_exposure(vol: pd.Series, target: float, cap: float = 1.0) -> pd.Series:
    """Volatility targeting : exposition = cible / volatilité réalisée, plafonnée.

    C'est le benchmark naturel d'une stratégie de régimes : réduire l'exposition
    quand la volatilité monte, sans aucun modèle.
    """
    return (target / vol).clip(upper=cap)


def calibrate_vol_target(vol: pd.Series, r: pd.Series, realized_target: float,
                         cap: float = 1.0, cost: float = 0.0005,
                         n_iter: int = 50) -> float:
    """Cible de volatilité qui donne, après backtest, la volatilité réalisée voulue.

    Sert à comparer le volatility targeting et une stratégie de régimes à risque égal.
    Recherche par dichotomie : la volatilité réalisée croît avec la cible.
    """
    low, high = 0.01, 1.0
    for _ in range(n_iter):
        mid = (low + high) / 2
        rets = backtest(vol_target_exposure(vol, mid, cap), r, cost)
        if rets.std() * np.sqrt(252) < realized_target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def annual_returns(results: dict) -> pd.DataFrame:
    """Performance année par année de chaque stratégie."""
    return pd.DataFrame({
        name: (1 + rets).groupby(rets.index.year).prod() - 1
        for name, rets in results.items()
    })
