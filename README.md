# Hydro-Alpha : USGS Streamflow → IDACORP Excess Return

Strategie quantitative qui predit le rendement excedentaire de l'action IDACORP (IDA) par rapport au secteur utilities (XLU) en exploitant des donnees hydrologiques, meteorologiques et energetiques publiques.

**Hypothese** : le debit des rivieres du Pacific Northwest impacte directement la production hydroelectrique d'IDACORP, et donc ses marges. En detectant des anomalies de debit (secheresses, crues), on peut anticiper la surperformance ou sous-performance du titre.

## Sources de donnees

Toutes les donnees sont **publiques et gratuites** :

| Source | Donnees | Frequence | Periode |
|--------|---------|-----------|---------|
| [USGS NWIS](https://waterservices.usgs.gov/) | Debit journalier de 4 rivieres (Columbia, Snake, Willamette, Deschutes) | Journalier | 2000 - present |
| [Yahoo Finance](https://finance.yahoo.com/) | Prix ajustes IDA et XLU | Journalier (trading) | 2000 - present |
| [ICE](https://www.ice.com/) | Prix spot electricite MID-C Hub (fichiers Excel locaux) | Journalier | 2001 - 2025 |
| [USDA SNOTEL](https://wcc.sc.egov.usda.gov/) | Snow Water Equivalent (SWE) - 4 stations Idaho | Journalier | 2000 - present |
| [Yahoo Finance](https://finance.yahoo.com/) | Prix futures gaz naturel Henry Hub (NG=F) | Journalier | 2000 - present |

## Structure du projet

```
ml-poc-project/
├── data/
│   ├── raw/hydro/              # Donnees brutes (generees par fetch_data.py)
│   │   ├── usgs_streamflow_daily.csv
│   │   ├── stock_prices_daily.csv
│   │   ├── ice_midc_daily.csv
│   │   ├── snotel_swe_daily.csv
│   │   └── henry_hub_gas_daily.csv
│   └── ice_electric-*.xls*     # Fichiers Excel ICE (inclus dans le repo)
├── models/                     # Modeles entraines (generes par train.py)
├── results/
│   ├── best_params.json        # Hyperparametres optimaux
│   ├── walk_forward_cv.csv     # Resultats walk-forward CV
│   └── model_metrics.csv       # Metriques finales par modele
├── plots/                      # Visualisations
├── notebooks/                  # Analyses exploratoires
├── scripts/
│   ├── fetch_data.py           # Telecharge toutes les donnees
│   ├── train.py                # Pipeline d'entrainement complet
│   └── main.py                 # Point d'entree : evaluation + Streamlit
├── src/
│   ├── config.py               # Configuration centrale
│   ├── data.py                 # Feature engineering + chargement
│   ├── metrics.py              # Metriques financieres et regression
│   ├── model_io.py             # Chargement des modeles
│   ├── app.py                  # Application Streamlit
│   └── results.py              # Ecriture des resultats
└── tests/                      # Tests unitaires
```

## Quickstart

### 1. Installer les dependances

```bash
git clone https://github.com/adelattre-cmd/ml-poc-project.git
cd ml-poc-project
pip install -r requirements.txt
```

Dependances supplementaires pour les fichiers Excel ICE :
```bash
pip install xlrd openpyxl
```

### 2. Telecharger les donnees

```bash
python scripts/fetch_data.py
```

Ce script telecharge les 5 sources de donnees et les sauvegarde dans `data/raw/hydro/`. Les fichiers Excel ICE sont deja inclus dans le repo, le script les parse automatiquement.

> Les donnees raw sont dans le `.gitignore` (trop volumineuses). Il faut lancer `fetch_data.py` apres chaque clone.

### 3. Entrainer les modeles

```bash
python scripts/train.py
```

Le pipeline d'entrainement execute 5 etapes :
1. **Feature pruning** — elimine les features non-informatives via Random Forest importance
2. **Hyperparameter tuning** — optimisation walk-forward sur 3 folds
3. **Walk-forward CV** — evaluation sur 5 folds (2001-2023)
4. **Entrainement final** — train <= 2018, test >= 2019
5. **Ensemble** — moyenne ponderee par IC des 4 modeles de base

Les modeles sont sauvegardes dans `models/`.

### 4. Lancer l'application

```bash
python scripts/main.py
```

Ce script :
- Verifie que les modeles existent (les entraine sinon)
- Evalue chaque modele sur le jeu de test
- Sauvegarde les metriques dans `results/model_metrics.csv`
- Lance l'application Streamlit sur `http://localhost:8501`

Options :
```bash
python scripts/main.py --force-train    # Re-entraine avant evaluation
python scripts/main.py --skip-train     # Echoue si les modeles manquent
```

### 5. Lancer les tests

```bash
PYTHONPATH=./src pytest tests/ -v
```

## Modeles

| Modele | Description |
|--------|-------------|
| Ridge | Baseline lineaire, coefficients interpretables |
| PCA + Ridge | Reduction PCA (90% variance) puis Ridge, reduit la multicolinearite |
| Random Forest | 300 arbres, capture les interactions non-lineaires |
| XGBoost | Gradient boosting avec early stopping, detecte les effets de seuil |
| Ensemble | Moyenne ponderee par IC walk-forward des 4 modeles de base |

## Feature engineering

**42 features brutes** construites, reduites a **~18 apres pruning** :

- **Debit des rivieres** (par riviere) : z-score saisonnier, percentile, tendance 30j, deficit cumule 90j, z-scores retardes (7j, 14j)
- **Interactions inter-rivieres** : ratio Snake/Columbia, pire z-score, deficit moyen, etendue de la secheresse
- **Snowpack SNOTEL** : z-score SWE, percentile, tendance, deficit cumule
- **Gaz naturel** : z-score prix, volatilite 30j, tendance
- **Interaction snowpack x gaz** : capture le scenario "double squeeze" (peu d'eau + gaz cher)
- **Saisonnalite** : sin/cos de la semaine
- **Momentum action** : rendement 20j IDA, momentum relatif IDA-XLU

## Metriques

| Metrique | Description |
|----------|-------------|
| IC | Information Coefficient (correlation de rang Spearman prediction vs realise) |
| Hit Rate | Proportion de directions correctement predites |
| Sharpe | Ratio de Sharpe annualise d'un portefeuille long/short |
| Gated Sharpe | Sharpe sur les predictions a haute conviction uniquement |
| RMSE / MAE / R2 | Metriques de regression standard |

## Configuration

Les parametres principaux sont dans `src/config.py` :

- `FORWARD_DAYS = 25` — horizon de prediction (jours de trading)
- `TARGET_TICKER = "IDA"` — action cible
- `BENCH_TICKER = "XLU"` — benchmark secteur utilities

Variable d'environnement optionnelle :
- `USE_ICE_FEATURES=1` — active les features electricite ICE (desactive par defaut car redondant avec les z-scores de debit)
