# Détection de régimes de marché par HMM

Détection en temps réel des régimes du S&P 500 (calme, tendu, nervosité, crise) avec un Hidden Markov Model gaussien, puis backtest d'une stratégie qui ajuste son exposition selon le régime.

L'accent est mis sur l'**absence de biais d'anticipation** : probabilités filtrées (algorithme forward) au lieu du décodage de Viterbi, réentraînement annuel en walk-forward, standardisation ajustée sur le passé seulement, décision à la clôture et exécution le lendemain.

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt
```

Dans VS Code, ouvre le dossier `regime-detection`, puis choisis l'interpréteur `.venv` comme kernel des notebooks.

## Utilisation

Lance les notebooks dans l'ordre. Chacun sauvegarde ce dont le suivant a besoin dans `data/`.

| Notebook | Contenu | Produit |
|---|---|---|
| `01_donnees_features` | Téléchargement du S&P 500 et du VIX, features | `data/features.parquet` |
| `02_hmm_2_etats` | Premier HMM sur les rendements seuls | — |
| `03_hmm_multivarie` | HMM sur `ret`, `vol_21`, `log(VIX)`, choix du nombre d'états par BIC, probabilités filtrées vs lissées | — |
| `04_walkforward` | Réentraînement annuel sans données futures | `data/probs_walkforward.parquet` |
| `05_backtest` | Stratégies, benchmarks, coûts de transaction | — |

## Structure

```
regime-detection/
├── notebooks/          # exploration et résultats
├── src/
│   ├── data.py         # téléchargement yfinance + cache CSV
│   ├── features.py     # log-rendements, volatilité, momentum, drawdown
│   ├── models.py       # HMM, ordre des états, filtre forward
│   ├── walkforward.py  # boucle d'entraînement walk-forward
│   └── backtest.py     # backtest, métriques, expositions, vol targeting
├── data/               # généré par les notebooks (non versionné)
└── requirements.txt
```

## Méthode

**Features.** Log-rendement journalier, volatilité réalisée sur 21 jours annualisée, log du VIX. Elles sont standardisées avant le HMM.

**Modèle.** HMM gaussien à 4 états et covariance pleine (`hmmlearn`). Le nombre d'états est choisi par BIC et par interprétabilité. Les états sont réordonnés par volatilité croissante pour régler le *label switching*. La matrice de transition a une structure en échelle : calme ↔ tendu ↔ nervosité ↔ crise, presque sans saut direct.

**Sans biais d'anticipation.**
1. Probabilités **filtrées** P(état_t | x_1..x_t), calculées par l'algorithme forward (`src/models.py`), et non `predict()` ou `predict_proba()`, qui utilisent toute la série.
2. **Walk-forward** : pour chaque année depuis 2010, le scaler et le HMM sont ajustés uniquement sur les années précédentes.
3. L'exposition décidée à la clôture de *t* s'applique au rendement de *t + 1*. Chaque changement d'exposition coûte 5 pb.

**Benchmarks.** Buy & hold, exposition constante égale à l'exposition moyenne de la stratégie, et **volatility targeting** calibré pour avoir la même volatilité réalisée que la stratégie de régimes.

## Résultats (2010-01-04 → 2026-11-17, walk-forward, coûts de 5 pb)

| Stratégie | Rendement annualisé | Volatilité | Sharpe | Max drawdown | Calmar | Exposition moyenne |
|---|---|---|---|---|---|---|
| Buy & hold | 11,7 % | 17,2 % | 0,68 | −33,9 % | 0,35 | 100 % |
| Régimes – binaire | 9,4 % | 13,7 % | 0,69 | −23,2 % | 0,41 | 94 % |
| Régimes – proportionnelle | 9,2 % | 13,7 % | 0,68 | −25,0 % | 0,37 | 94 % |
| Régimes – graduée | 8,6 % | 12,0 % | 0,72 | −18,6 % | 0,46 | 87 % |
| Constante 94 % | 11,0 % | 16,1 % | 0,69 | −32,1 % | 0,34 | 94 % |
| Vol target (même vol que graduée) | 9,3 % | 12,0 % | 0,77 | −16,7 % | 0,55 | 86 % |

- Les stratégies de régimes divisent à peu près par deux le max drawdown (−34 % → −19 % pour la graduée), mais coûtent 2,3 à 3,1 % de rendement annualisé face au buy & hold.
- À volatilité égale, le simple volatility targeting fait un peu mieux que la stratégie graduée (Sharpe 0,77 contre 0,72) : le HMM n'apporte pas de gain net par rapport à ce benchmark.
- Le régime walk-forward et le régime du modèle entraîné sur toute la période concordent sur 92,5 % des jours.
- La sous-performance se concentre sur les crises en V (2011, 2018, 2020, 2025) : le modèle reste défensif pendant le rebond, le temps que la volatilité retombe.
- 2022, baisse lente sans krach, est le cas où le modèle apporte le plus.

## Limites et pistes

- La volatilité réalisée réagit avec retard : le modèle détecte bien l'entrée en crise, mais tarde à en sortir.
- Pistes : *Statistical Jump Models* (Nystrup et al.) pour des régimes plus stables, features de crédit et de structure par terme du VIX, extension multi-actifs.
