"""HMM gaussiens : entraînement, ordre des états et probabilités filtrées."""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

STATE_NAMES = ["calme", "tendu", "nervosite", "crise"]


def fit_hmm(X: np.ndarray, n_states: int, random_state: int = 42,
            n_iter: int = 1000) -> GaussianHMM:
    """Entraîne un HMM gaussien à covariance pleine."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    return model


def fit_best_hmm(X: np.ndarray, n_states: int, n_seeds: int = 5) -> GaussianHMM:
    """Entraîne `n_seeds` HMM avec des initialisations différentes et garde le meilleur.

    Chaque entraînement peut tomber dans un optimum local différent : on garde celui
    qui a la plus grande log-vraisemblance.
    """
    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        model = fit_hmm(X, n_states, random_state=seed)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def state_order(model: GaussianHMM, feature: int = 1) -> np.ndarray:
    """Ordre des états par moyenne croissante de la feature `feature`.

    Avec feature=1 (vol_21), on obtient l'ordre calme -> crise. Ça règle le
    « label switching » : la numérotation brute de hmmlearn est arbitraire.
    """
    return np.argsort(model.means_[:, feature])


def order_states(model: GaussianHMM, states: np.ndarray, feature: int = 1):
    """Renumérote une séquence d'états du calme à la crise.

    Renvoie (états renumérotés, matrice de transition réordonnée).
    """
    order = state_order(model, feature)
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))
    return mapping[states], model.transmat_[np.ix_(order, order)]


def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """Durée moyenne de chaque régime, en jours : 1 / (1 - p_ii)."""
    return 1 / (1 - np.diag(transmat))


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Répartition de long terme impliquée par la matrice de transition."""
    eigvals, eigvecs = np.linalg.eig(transmat.T)
    pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    return pi / pi.sum()


def filtered_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Probabilités filtrées P(état_t | x_1..x_t), shape (n_jours, n_etats).

    Algorithme forward : chaque jour, on prédit avec la matrice de transition,
    on corrige avec la vraisemblance de l'observation du jour, puis on normalise.
    Contrairement à predict() et predict_proba(), aucune donnée future n'est utilisée.

    Utilise la numérotation d'origine du modèle : appliquer [:, state_order(model)]
    pour passer dans l'ordre calme -> crise.
    """
    n_states = model.n_components
    T = model.transmat_

    # Log-vraisemblances, shape (n_jours, n_etats). On passe par le log et on retire
    # le max de chaque ligne pour éviter que les jours extrêmes donnent 0 partout.
    log_b = np.column_stack([
        multivariate_normal(model.means_[k], model.covars_[k]).logpdf(X)
        for k in range(n_states)
    ])
    B = np.exp(log_b - log_b.max(axis=1, keepdims=True))

    n_days = B.shape[0]
    alpha = np.zeros_like(B)

    v = model.startprob_ * B[0]
    alpha[0] = v / v.sum()

    for t in range(1, n_days):
        prediction = alpha[t - 1] @ T
        v = prediction * B[t]
        total = v.sum()
        # Garde-fou : si la prédiction exclut tous les états plausibles, on repart
        # de la seule vraisemblance du jour
        alpha[t] = v / total if total > 0 else B[t] / B[t].sum()

    return alpha
