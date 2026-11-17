"""Entraînement walk-forward : le modèle n'utilise jamais de données futures."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models import STATE_NAMES, filtered_probs, fit_best_hmm, state_order


def walk_forward(feats: pd.DataFrame, first_test_year: int = 2010,
                 last_test_year: int | None = None, n_states: int = 4,
                 n_seeds: int = 3, order_feature: int = 1,
                 state_names: list[str] = STATE_NAMES, verbose: bool = True):
    """Probabilités de régime filtrées, réentraînées chaque année.

    Pour chaque année de test :
    1. on ne garde que les données jusqu'au 31 décembre de l'année précédente ;
    2. on ajuste le StandardScaler sur ce passé seulement ;
    3. on entraîne un HMM et on ordonne ses états du calme à la crise ;
    4. on filtre jour par jour jusqu'à la fin de l'année de test, et on garde
       seulement les lignes de l'année de test.

    Renvoie (DataFrame des probabilités, dict année -> (scaler, modèle, ordre)).
    """
    if last_test_year is None:
        last_test_year = feats.index[-1].year

    blocks, fitted = [], {}

    for year in range(first_test_year, last_test_year + 1):
        train = feats.loc[: f"{year - 1}-12-31"]
        history = feats.loc[: f"{year}-12-31"]
        if (history.index.year == year).sum() == 0:
            continue

        scaler = StandardScaler().fit(train)
        model = fit_best_hmm(scaler.transform(train), n_states, n_seeds=n_seeds)
        order = state_order(model, order_feature)

        alpha = filtered_probs(model, scaler.transform(history))[:, order]
        probs = pd.DataFrame(alpha, index=history.index, columns=state_names[:n_states])
        blocks.append(probs.loc[str(year)])
        fitted[year] = (scaler, model, order)

        if verbose:
            print(f"{year} : entraîné sur {len(train)} jours, "
                  f"{len(blocks[-1])} jours filtrés")

    return pd.concat(blocks), fitted
