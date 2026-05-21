# Hydro-Alpha : Signal Hydrologique → Alpha sur Utilities PNW

Strategie quantitative qui predit le rendement excedentaire des utilities hydroelectriques du Pacific Northwest (IDA, AVA, POR) par rapport au secteur utilities (XLU) en exploitant des donnees hydrologiques, climatiques et energetiques publiques.

**Hypothese** : le debit des rivieres du PNW impacte directement la production hydroelectrique d'IDACORP, Avista et Portland General. En detectant des anomalies de debit, de snowpack et de regime ENSO, on anticipe la sur/sous-performance de ces titres.

**Chaine causale** :
```
El Nino (ENSO+) → hiver sec dans le PNW → faible manteau neigeux
    → debit fluvial bas → achat d'electricite spot chere
    → compression des marges → sous-performance vs XLU
```

## Sources de donnees

Toutes les donnees sont **publiques et gratuites** :

| Source | Donnees | Frequence | Periode |
|--------|---------|-----------|---------|
| [USGS NWIS](https://waterservices.usgs.gov/) | Debit journalier de 4 rivieres (Columbia, Snake, Willamette, Deschutes) | Journalier | 2000 - present |
| [Yahoo Finance](https://finance.yahoo.com/) | Prix ajustes IDA, AVA, POR et XLU | Journalier (trading) | 2000 - present |
| [ICE](https://www.ice.com/) | Prix spot electricite MID-C Hub (fichiers Excel locaux) | Journalier | 2001 - 2025 |
| [USDA SNOTEL](https://wcc.sc.egov.usda.gov/) | Snow Water Equivalent (SWE) - 4 stations Idaho | Journalier | 2000 - present |
| [Yahoo Finance](https://finance.yahoo.com/) | Prix futures gaz naturel Henry Hub (NG=F) | Journalier | 2000 - present |
| [NOAA CPC](https://www.cpc.ncep.noaa.gov/) | Indice ONI (El Nino / La Nina) | Mensuel | 1950 - present |

## Structure du projet

```
ml-poc-project/
├── data/
│   ├── raw/hydro/              # Donnees brutes (generees par fetch_data.py)
│   │   ├── usgs_streamflow_daily.csv
│   │   ├── stock_prices_daily.csv
│   │   ├── ice_midc_daily.csv
│   │   ├── snotel_swe_daily.csv
│   │   ├── henry_hub_gas_daily.csv
│   │   └── enso_oni_daily.csv
│   └── ice_electric-*.xls*     # Fichiers Excel ICE (inclus dans le repo)
├── models/                     # Modeles entraines (generes par train.py)
├── results/
│   ├── best_params.json        # Hyperparametres optimaux
│   ├── selected_features.json  # Features retenues apres pruning
│   ├── walk_forward_cv.csv     # Resultats walk-forward CV
│   └── model_metrics.csv       # Metriques finales par modele
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
pip install xlrd openpyxl   # pour les fichiers Excel ICE
```

### 2. Telecharger les donnees

```bash
python scripts/fetch_data.py
```

Telecharge 6 sources de donnees dans `data/raw/hydro/`. L'indice ENSO est telecharge depuis NOAA (pas de cle API requise). Les fichiers Excel ICE sont deja dans le repo.

> Les donnees raw sont dans le `.gitignore`. Il faut lancer `fetch_data.py` apres chaque clone.

### 3. Entrainer les modeles

```bash
python scripts/train.py
```

Pipeline d'entrainement :
1. **Feature pruning** — elimine les features non-informatives (Random Forest importance, seuil 0.01)
2. **Hyperparameter tuning** — walk-forward sur 3 folds
3. **Walk-forward CV** — 5 folds (2001-2023)
4. **Entrainement final** — train <= 2018, test >= 2019
5. **Ensemble** — moyenne ponderee par IC

### 4. Lancer l'application

```bash
python scripts/main.py
```

Lance l'evaluation des modeles et l'application Streamlit sur `http://localhost:8501`.

Options :
```bash
python scripts/main.py --force-train    # re-entraine avant evaluation
python scripts/main.py --skip-train     # echoue si modeles manquent
```

### 5. Lancer les tests

```bash
PYTHONPATH=./src pytest tests/ -v
```

## Modeles

| Modele | Description |
|--------|-------------|
| Ridge | Baseline lineaire, coefficients interpretables |
| PCA + Ridge | PCA (90% variance) + Ridge, reduit la multicolinearite |
| **Random Forest** | **300 arbres, meilleur IC en test (+0.20), Sharpe +0.58** |
| XGBoost | Gradient boosting, detecte les effets de seuil |
| Ensemble | Moyenne ponderee par IC des 4 modeles |

## Feature engineering

**48 features brutes** construites, reduites a **~18 apres pruning** :

- **Debit des rivieres** : z-score saisonnier, percentile, tendance 30j, deficit cumule 90j, z-scores retardes (7j, 14j)
- **Interactions inter-rivieres** : ratio Snake/Columbia, pire z-score, deficit moyen, etendue secheresse
- **Snowpack SNOTEL** : z-score SWE, percentile, tendance, deficit cumule
- **ENSO (ONI)** : indice ONI, valeur absolue, tendance 3 mois, interaction ONI x snowpack
- **Gaz naturel** : z-score prix, volatilite 30j, tendance
- **Interactions** : snowpack x gaz ("double squeeze"), ONI x snowpack
- **Saisonnalite** : sin/cos de la semaine
- **Momentum** : rendement 20j IDA, momentum relatif IDA-XLU

## Backtest ameliore

Le backtest applique le signal a un portfolio de 3 utilities hydro PNW (IDA, AVA, POR) avec :

- **Regime filter** : ne trade que quand le signal depasse sa mediane historique (haute conviction) — double le rendement (+37% vs +20%) en eliminant les trades sans conviction
- **Signal transferable** : le modele (entraine sur IDA) a un IC positif sur les 3 titres — AVA repond particulierement bien au signal

> Le vol targeting (ajustement par volatilite) a ete teste et ecarte : il detruisait les rendements car les periodes volatiles sont celles ou le signal hydro est le plus fort.

## Metriques

| Metrique | Description |
|----------|-------------|
| IC | Information Coefficient (correlation de rang Spearman prediction vs realise) |
| Hit Rate | Proportion de directions correctement predites |
| Sharpe | Ratio de Sharpe annualise d'un portefeuille long/short |
| Gated Sharpe | Sharpe sur les predictions a haute conviction uniquement |
| RMSE / MAE / R2 | Metriques de regression standard |

## Configuration

Parametres dans `src/config.py` :

- `FORWARD_DAYS = 25` — horizon de prediction (jours de trading)
- `TARGET_TICKER = "IDA"` — action cible pour l'entrainement
- `BENCH_TICKER = "XLU"` — benchmark secteur utilities

Variable d'environnement optionnelle :
- `USE_ICE_FEATURES=1` — active les features electricite ICE (desactive par defaut)
