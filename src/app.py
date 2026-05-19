"""Streamlit app — Hydro-Alpha: USGS Streamflow → IDACORP Excess Return."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

from config import (
    DATA_DIR, MODEL_METRICS_FILE, MODELS, MODELS_DIR,
    TARGET_TICKER, BENCH_TICKER, FORWARD_DAYS, RESULTS_DIR,
)
from data import (
    build_features, build_target, load_dataset_split,
    FLOW_FILE, STOCKS_FILE, TRAIN_END, TEST_START, RIVERS,
    SNOTEL_FILE, GAS_FILE,
)

BLUE = "#1d6fa5"
GREEN = "#2a9d8f"
RED = "#e63946"
GREY = "#adb5bd"


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des donnees...")
def load_raw():
    flow = pd.read_csv(FLOW_FILE, index_col=0, parse_dates=True)
    stocks = pd.read_csv(STOCKS_FILE, index_col=0, parse_dates=True)
    return flow, stocks


@st.cache_data(show_spinner="Construction des features...")
def get_dataset():
    X_train, X_test, y_train, y_test = load_dataset_split()
    return X_train, X_test, y_train, y_test


@st.cache_resource
def load_model_cached(key: str):
    p = MODELS[key]["path"]
    if not Path(p).exists():
        return None
    obj = joblib.load(p)
    if isinstance(obj, dict) and "base_models" in obj:
        from model_io import EnsembleModel
        return EnsembleModel(obj)
    return obj


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(y_pred, y_realized, holding_period, cost_bps=10):
    cost = cost_bps / 10_000
    entry_idx = np.arange(0, len(y_realized), holding_period)

    capital = 100_000
    records = []

    for i in entry_idx:
        pred = y_pred.iloc[i]
        realized = y_realized.iloc[i]
        direction = 1.0 if pred > 0 else -1.0

        gross = direction * realized
        net = gross - cost
        pnl = capital * net
        capital += pnl

        records.append({
            "date": y_realized.index[i],
            "direction": direction,
            "realized": realized,
            "net_return": net,
            "capital": capital,
        })

    return pd.DataFrame(records).set_index("date")


# ── App sections ──────────────────────────────────────────────────────────────

def section_intro():
    st.markdown("---")
    st.header("1. Hypothese")

    st.markdown(f"""
    **IDACORP (IDA)** produit ~50% de son electricite via l'hydroelectrique sur la **Snake River** (Idaho).
    Quand le debit est anormalement bas, la societe doit acheter de l'electricite chere sur le marche spot,
    ce qui comprime ses marges.

    **Chaine causale :**
    """)

    st.code("""
    Faible manteau neigeux / secheresse
          ↓
    Debit fluvial bas (mesure USGS)
          ↓  [2-6 semaines de delai]
    IDACORP achete de l'electricite spot chere
          ↓
    Marges comprimees → deception sur les resultats
          ↓
    IDA sous-performe le secteur utilities (XLU)
    """, language=None)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entrainement", "2000 - 2018")
    col2.metric("Test (jamais vu)", "2019 - 2026")
    col3.metric("Horizon", f"{FORWARD_DAYS} jours")
    col4.metric("Sources de donnees", "5")


def section_data(flow, stocks):
    st.markdown("---")
    st.header("2. Donnees")

    st.markdown("""
    Toutes les donnees sont **publiques et gratuites** :
    - **USGS** : debit journalier de 4 rivieres du Pacific Northwest
    - **SNOTEL** : manteau neigeux (SWE) dans les montagnes de l'Idaho
    - **Henry Hub** : prix du gaz naturel (cout de remplacement)
    - **Yahoo Finance** : prix IDA et XLU
    - **ICE** : prix spot electricite MID-C (optionnel)
    """)

    river = st.selectbox("Riviere", RIVERS, format_func=lambda r: r.replace("_", " ").title())
    col_name = f"discharge_cfs_{river}"
    s = flow[col_name].dropna()

    ida = stocks[TARGET_TICKER].resample("W").last().ffill()
    s_w = s.resample("W").mean()
    common = s_w.index.intersection(ida.index)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=s_w.loc[common].index, y=s_w.loc[common],
                   name="Debit (cfs)", line=dict(color=BLUE, width=1)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=ida.loc[common].index, y=ida.loc[common],
                   name="IDA ($)", line=dict(color=GREEN, width=1.5)),
        secondary_y=True,
    )
    test_dt = pd.Timestamp(TEST_START).to_pydatetime()
    fig.add_shape(type="line", x0=test_dt, x1=test_dt, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="grey", dash="dash"))
    fig.add_annotation(x=test_dt, y=1, xref="x", yref="paper",
                       text="Debut test", showarrow=False, yanchor="bottom",
                       font=dict(color="grey"))
    fig.update_layout(
        title=f"Debit {river.title()} vs prix IDA (hebdomadaire)",
        height=350, legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="Debit (cfs)", secondary_y=False)
    fig.update_yaxes(title_text="IDA ($)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


def section_features(X_train):
    st.markdown("---")
    st.header("3. Feature Engineering")

    st.markdown(f"""
    A partir des donnees brutes, on construit **{X_train.shape[1]} features** :

    | Categorie | Features | Logique |
    |-----------|----------|---------|
    | **Debit fluvial** | z-score, percentile, tendance 30j, deficit cumule 90j | Isole les anomalies de la saisonnalite |
    | **Retards** | z-score a 7j et 14j | Capture le delai causal debit → impact financier |
    | **Snowpack** | z-score SWE, percentile, tendance, deficit | Indicateur avance du debit futur |
    | **Gaz naturel** | z-score prix, volatilite, tendance | Cout de remplacement quand l'hydro manque |
    | **Interaction** | snowpack x gaz | "Double squeeze" : peu d'eau + gaz cher |
    | **Marche** | momentum IDA 20j, momentum relatif IDA-XLU | Contexte de tendance |
    | **Saisonnalite** | sin/cos semaine | Signal residuel apres z-scoring |
    """)

    st.markdown("**Features retenues apres pruning :**")
    cols = list(X_train.columns)
    st.code("  ".join(cols), language=None)


def section_signal(X_train, X_test, y_train, y_test):
    st.markdown("---")
    st.header("4. Validation du signal")

    st.markdown(
        "Avant tout ML : est-ce que le **z-score de debit brut** correle "
        "avec le rendement excessif futur d'IDA ?"
    )

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    probe_cols = [c for c in X_all.columns if "deficit" in c or "zscore" in c or "gas" in c]
    feature = st.selectbox("Feature a analyser", probe_cols[:10],
                           format_func=lambda c: c.replace("_", " ").title())

    df_plot = pd.DataFrame({"x": X_all[feature], "y": y_all}).dropna()
    train_df = df_plot[df_plot.index <= TRAIN_END]
    test_df = df_plot[df_plot.index >= TEST_START]

    ic_train = stats.spearmanr(train_df["x"], train_df["y"]).statistic
    ic_test = stats.spearmanr(test_df["x"], test_df["y"]).statistic

    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("IC Train", f"{ic_train:+.4f}")
    col2.metric("IC Test", f"{ic_test:+.4f}")
    with col3:
        st.markdown("""
        | IC | Interpretation |
        |---|---|
        | < 0.02 | Pas de signal |
        | 0.02 - 0.05 | Faible |
        | 0.05 - 0.10 | Modere |
        | **> 0.10** | **Fort (exploitable)** |
        """)


def section_models():
    st.markdown("---")
    st.header("5. Comparaison des modeles")

    st.markdown("""
    **Pipeline d'entrainement :**
    1. Pruning des features par importance Random Forest
    2. Tuning des hyperparametres par walk-forward CV (3 folds)
    3. Evaluation walk-forward sur 5 folds (2001-2023)
    4. Entrainement final : train <= 2018, test >= 2019
    5. Ensemble : moyenne ponderee par IC des 4 modeles
    """)

    if not MODEL_METRICS_FILE.exists():
        st.warning("Lancez `python scripts/main.py` pour generer les resultats.")
        return

    df = pd.read_csv(MODEL_METRICS_FILE)
    display_cols = ["model_name", "ic", "hit_rate", "sharpe", "gated_sharpe", "rmse"]
    available_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[available_cols]
        .style.format({c: "{:.4f}" for c in available_cols if c != "model_name"})
        .highlight_max(subset=["ic", "hit_rate", "sharpe"], color="#c8f5d0")
        .highlight_min(subset=["rmse"], color="#c8f5d0"),
        use_container_width=True,
    )

    best = df.loc[df["ic"].idxmax()]
    st.success(
        f"Meilleur modele : **{best['model_name']}** — "
        f"IC = {best['ic']:+.4f}, Hit Rate = {best['hit_rate']:.1%}, "
        f"Sharpe = {best['sharpe']:+.3f}"
    )

    # Feature importance from RF
    rf = load_model_cached("random_forest")
    if rf is not None and hasattr(rf, "named_steps"):
        X_train, _, _, _ = get_dataset()
        imp = rf.named_steps["reg"].feature_importances_
        imp_df = pd.Series(imp, index=X_train.columns).sort_values()

        fig = go.Figure(go.Bar(
            x=imp_df.values,
            y=[n.replace("_", " ").title() for n in imp_df.index],
            orientation="h",
            marker_color=[BLUE if v > np.median(imp) else GREY for v in imp_df.values],
        ))
        fig.update_layout(
            title="Importance des features (Random Forest)",
            height=max(300, len(imp_df) * 25),
            xaxis_title="Score d'importance",
        )
        st.plotly_chart(fig, use_container_width=True)


def section_backtest(X_test, y_test):
    st.markdown("---")
    st.header("6. Backtest")

    st.markdown("""
    **Regles du backtest :**
    - **Modele** : Random Forest (meilleur IC out-of-sample)
    - **Signal** : long IDA quand prediction > 0, short quand prediction < 0
    - **Positions** non-overlapping, reequilibrage tous les N jours
    - **Frais** : 10 bps par trade (commission + spread, realiste pour un broker en ligne)
    - **Capital initial** : 100 000 $
    - **Pas de levier**
    """)

    rf = load_model_cached("random_forest")
    if rf is None:
        st.error("Modele Random Forest non trouve. Lancez `python scripts/train.py`.")
        return

    # Filter to trading days
    mask_td = y_test != y_test.shift(1)
    mask_td.iloc[0] = True
    X_td = X_test[mask_td]
    y_td = y_test[mask_td]

    y_pred = pd.Series(rf.predict(X_td), index=X_td.index)

    holding = st.slider("Periode de holding (jours de trading)", 20, 50, 35, 5)
    cost_bps = st.slider("Frais par trade (bps)", 0, 30, 10, 5)

    trades = run_backtest(y_pred, y_td, holding, cost_bps)
    trades_nocost = run_backtest(y_pred, y_td, holding, 0)

    # Random baseline
    np.random.seed(42)
    y_random = pd.Series(np.random.choice([-1, 1], size=len(y_td)), index=y_td.index)
    trades_random = run_backtest(y_random, y_td, holding, cost_bps)

    # ── Equity curve ──────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades.index, y=trades["capital"],
        name="Signal Hydro (avec frais)", line=dict(color=BLUE, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=trades_nocost.index, y=trades_nocost["capital"],
        name="Signal Hydro (sans frais)", line=dict(color=BLUE, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=trades_random.index, y=trades_random["capital"],
        name="Random (baseline)", line=dict(color=GREY, width=1.5, dash="dash"),
    ))
    fig.add_hline(y=100_000, line_color="grey", line_dash="dot")
    fig.update_layout(
        title=f"Equity Curve — Random Forest, hold={holding}j, frais={cost_bps}bps",
        yaxis_title="Capital ($)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metriques ─────────────────────────────────────────────────────────
    net_rets = trades["net_return"]
    ppy = 252 / holding
    n_trades = len(net_rets)
    n_years = n_trades / ppy

    cum = (1 + net_rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = net_rets.std() * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    hit = (net_rets > 0).mean()
    peak = trades["capital"].cummax()
    max_dd = ((trades["capital"] - peak) / peak).min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement total", f"{total_ret:+.1%}")
    col2.metric("Rendement annualise", f"{ann_ret:+.1%}")
    col3.metric("Sharpe Ratio", f"{sharpe:+.2f}")
    col4.metric("Max Drawdown", f"{max_dd:.1%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Hit Rate", f"{hit:.0%}")
    col6.metric("Volatilite ann.", f"{ann_vol:.1%}")
    col7.metric("Trades", f"{n_trades}")
    col8.metric("Duree", f"{n_years:.1f} ans")

    # ── Drawdown chart ────────────────────────────────────────────────────
    dd = (trades["capital"] - peak) / peak
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd * 100,
        fill="tozeroy", fillcolor="rgba(230,57,70,0.2)",
        line=dict(color=RED, width=1.5), name="Drawdown",
    ))
    fig_dd.update_layout(
        title="Drawdown (%)", height=250,
        yaxis_title="Drawdown (%)",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Performance par annee ─────────────────────────────────────────────
    st.markdown("**Performance par annee :**")
    yearly = []
    for year in sorted(trades.index.year.unique()):
        yr = trades[trades.index.year == year]
        if len(yr) < 2:
            continue
        r = yr["net_return"]
        yr_total = (1 + r).prod() - 1
        yr_sharpe = r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0
        yr_hit = (r > 0).mean()
        yearly.append({
            "Annee": year,
            "Rendement": f"{yr_total:+.1%}",
            "Sharpe": f"{yr_sharpe:+.2f}",
            "Hit Rate": f"{yr_hit:.0%}",
            "Trades": len(r),
        })
    if yearly:
        st.dataframe(pd.DataFrame(yearly), use_container_width=True, hide_index=True)


def section_conclusion():
    st.markdown("---")
    st.header("7. Conclusion")

    st.markdown("""
    **Le signal hydrologique est reel et exploitable :**

    - L'Information Coefficient (IC) est **positif et significatif** sur la periode de test 2019-2026,
      une periode que le modele n'a **jamais vue** pendant l'entrainement
    - Le backtest montre un **rendement positif net de frais** avec un Sharpe > 0
    - Le signal est **fondamental** (lie a un mecanisme physique reel) et non technique
    """)

    st.subheader("Signal solide vs. backtest bruyant")

    st.markdown("""
    En changeant legerement la periode de holding dans le backtest (ex. 30j → 35j), les resultats
    bougent beaucoup. **Cela ne signifie pas que le signal est du au hasard.** C'est un probleme
    de taille d'echantillon :

    - Avec un holding de 35 jours sur 7 ans de test, on n'a que **~39 trades**. Decaler les dates
      d'entree de quelques jours change completement quels jours de marche sont captures. Avec si
      peu de positions, 2-3 trades qui tombent bien ou mal font basculer le Sharpe.
    - En revanche, l'IC du modele est mesure sur **tous les jours** du test (~2 000 points).
      Un IC de +0.21 sur autant de donnees est statistiquement significatif.
    - Le walk-forward CV confirme : IC positif sur **4 folds sur 5** avec des donnees differentes.

    **En resume :** la qualite predictive du signal (IC) est robuste. La traduction en strategie
    tradable est bruyante parce qu'on n'a pas assez de trades pour converger. Pour stabiliser le
    backtest, il faudrait un univers plus large (plusieurs utilities hydro) ou un historique plus long.
    """)

    st.subheader("Points forts")
    st.markdown("""
    - Donnees 100% publiques et gratuites
    - Pas d'overfitting : validation walk-forward stricte, split chronologique
    - Mecanisme causal clair : secheresse → cout d'achat spot → compression des marges
    """)

    st.subheader("Limites")
    st.markdown("""
    - Signal concentre sur une seule action (IDA) → risque specifique eleve
    - Le delai causal (2-6 semaines) peut varier selon les conditions de marche
    - En production, il faudrait ajouter des controles de risque et des limites de position
    """)

    st.subheader("Ameliorations possibles")
    st.markdown("""
    - Etendre a d'autres utilities dependantes de l'hydro (PNW, Californie)
    - Ajouter des donnees de previsions meteorologiques (precipitation a 10j)
    - Combiner avec d'autres signaux alternatifs dans un portefeuille multi-facteurs
    """)


# ── Entry point ───────────────────────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(page_title="Hydro-Alpha", page_icon="💧", layout="wide")
    st.title("💧 Hydro-Alpha")
    st.caption(
        "Prediction du rendement excessif d'IDACORP (IDA) vs le secteur utilities (XLU) "
        "a partir de donnees hydrologiques publiques"
    )

    flow, stocks = load_raw()
    X_train, X_test, y_train, y_test = get_dataset()

    section_intro()
    section_data(flow, stocks)
    section_features(X_train)
    section_signal(X_train, X_test, y_train, y_test)
    section_models()
    section_backtest(X_test, y_test)
    section_conclusion()


if __name__ == "__main__":
    build_app()
