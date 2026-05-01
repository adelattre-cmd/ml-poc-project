"""Streamlit app — Hydro-Alpha: USGS Streamflow → IDACORP Excess Return."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

from config import (
    DATA_DIR, MODEL_METRICS_FILE, MODELS, MODELS_DIR,
    TARGET_TICKER, BENCH_TICKER, FORWARD_DAYS,
)
from data import (
    build_features, build_target, load_dataset_split,
    FLOW_FILE, STOCKS_FILE, TRAIN_END, TEST_START, RIVERS,
)

HYDRO_COLOR  = "#1d6fa5"
RETURN_UP    = "#2a9d8f"
RETURN_DOWN  = "#e63946"
NEUTRAL      = "#adb5bd"


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_raw():
    flow   = pd.read_csv(FLOW_FILE,   index_col=0, parse_dates=True)
    stocks = pd.read_csv(STOCKS_FILE, index_col=0, parse_dates=True)
    return flow, stocks


@st.cache_data(show_spinner="Building features…")
def get_dataset():
    X_train, X_test, y_train, y_test = load_dataset_split()
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    return X_train, X_test, y_train, y_test, X_all, y_all


@st.cache_resource
def load_model(key: str):
    p = MODELS[key]["path"]
    return joblib.load(p) if Path(p).exists() else None


# ── Sections ───────────────────────────────────────────────────────────────────
def _overview():
    st.header("The Idea")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
        **IDACORP (IDA)** generates ~50 % of its electricity from hydropower on the
        **Snake River** (Idaho). When river flow is anomalously low, the company
        must buy expensive power on the spot market to meet demand — compressing margins.

        The hypothesis: **USGS streamflow anomalies predict IDA's excess return over
        the utilities sector (XLU) {FORWARD_DAYS} trading days ahead.**

        This is a real *alternative data* signal — the kind that quant funds pay
        millions for. Here it comes free from a government API.

        **Causal chain:**
        ```
        Low snowpack / drought
              ↓
        Low river discharge (USGS)
              ↓  [2-6 week lag]
        IDA buys expensive spot power
              ↓
        Compressed margins → earnings miss
              ↓
        IDA underperforms utilities sector (XLU)
        ```
        """)
    with col2:
        st.metric("Training period", f"2000 → 2018")
        st.metric("Test period (unseen)", f"2019 → 2025")
        st.metric("Prediction horizon", f"{FORWARD_DAYS} trading days")
        st.metric("Gauges", "4 rivers, daily since 2000")
        st.metric("Features", "20 (z-scores, percentiles, trends, deficits)")


def _streamflow(flow, stocks):
    st.header("Streamflow Data")

    rivers_display = {
        "columbia":   "Columbia River at The Dalles, OR",
        "snake":      "Snake River at Weiser, ID",
        "willamette": "Willamette River at Portland, OR",
        "deschutes":  "Deschutes River at Moody, OR",
    }

    river = st.selectbox("River gauge", list(rivers_display.keys()),
                         format_func=lambda k: rivers_display[k])
    col_name = f"discharge_cfs_{river}"
    s = flow[col_name].dropna()

    col1, col2 = st.columns(2)

    with col1:
        # Raw flow + IDA price dual-axis
        ida = stocks[TARGET_TICKER].resample("W").last().ffill()
        s_w = s.resample("W").mean()
        common = s_w.index.intersection(ida.index)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=s_w.loc[common].index, y=s_w.loc[common],
                                 name="Discharge (cfs)", line=dict(color=HYDRO_COLOR, width=1)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=ida.loc[common].index, y=ida.loc[common],
                                 name="IDA price ($)", line=dict(color=RETURN_UP, width=1.5)),
                      secondary_y=True)
        fig.add_vline(x=TEST_START, line_dash="dash", line_color="grey",
                      annotation_text="Test start")
        fig.update_layout(title=f"{rivers_display[river]} vs IDA (weekly)",
                          height=350, legend=dict(orientation="h"))
        fig.update_yaxes(title_text="Discharge (cfs)", secondary_y=False)
        fig.update_yaxes(title_text="IDA ($)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Seasonal profile
        week = s.index.isocalendar().week.astype(int)
        seasonal = s.groupby(week).agg(["mean", "std"])
        fig2 = go.Figure([
            go.Scatter(x=seasonal.index,
                       y=seasonal["mean"] + seasonal["std"],
                       fill=None, mode="lines", line_color="lightblue",
                       showlegend=False),
            go.Scatter(x=seasonal.index,
                       y=seasonal["mean"] - seasonal["std"],
                       fill="tonexty", mode="lines", line_color="lightblue",
                       name="±1 std", fillcolor="rgba(29,111,165,0.15)"),
            go.Scatter(x=seasonal.index, y=seasonal["mean"],
                       mode="lines", name="Mean",
                       line=dict(color=HYDRO_COLOR, width=2)),
        ])
        fig2.update_layout(title="Seasonal profile (all years)",
                           xaxis_title="Week of year",
                           yaxis_title="Mean discharge (cfs)",
                           height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "The seasonal cycle is strong (snowmelt peaks in May-June). "
        "The model uses **z-scores vs the historical weekly mean** so it sees "
        "pure anomalies — not just 'it is spring'."
    )


def _signal_analysis(X_all, y_all):
    st.header("Signal Analysis")
    st.markdown(
        "Before any ML: does the **raw streamflow z-score** correlate with "
        "forward IDA excess returns? If yes, we have a genuine signal to model."
    )

    feature = st.selectbox(
        "Feature",
        [c for c in X_all.columns if "zscore" in c or "deficit" in c or "pct" in c],
        format_func=lambda c: c.replace("_", " ").title(),
    )

    df_plot = pd.DataFrame({"x": X_all[feature], "y": y_all,
                            "period": np.where(X_all.index < TEST_START, "Train", "Test")})
    df_plot = df_plot.dropna()

    col1, col2 = st.columns(2)
    with col1:
        ic_train = stats.spearmanr(
            df_plot[df_plot.period == "Train"]["x"],
            df_plot[df_plot.period == "Train"]["y"]
        ).statistic
        ic_test  = stats.spearmanr(
            df_plot[df_plot.period == "Test"]["x"],
            df_plot[df_plot.period == "Test"]["y"]
        ).statistic

        fig = px.scatter(df_plot, x="x", y="y", color="period",
                         color_discrete_map={"Train": NEUTRAL, "Test": HYDRO_COLOR},
                         opacity=0.3, trendline="ols",
                         labels={"x": feature, "y": f"IDA excess return (fwd {FORWARD_DAYS}d)"},
                         title=f"Scatter: {feature} vs forward excess return")
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("IC — Train", f"{ic_train:+.4f}")
        st.metric("IC — Test (unseen)", f"{ic_test:+.4f}",
                  delta=f"{ic_test - ic_train:+.4f} vs train")
        st.markdown("""
        **IC interpretation:**
        | IC | Meaning |
        |---|---|
        | < 0.02 | No signal |
        | 0.02 – 0.05 | Weak |
        | 0.05 – 0.10 | Moderate |
        | > 0.10 | **Strong (usable)** |
        """)

    # IC by calendar month
    df_plot["month"] = pd.to_datetime(df_plot.index).month
    monthly_ic = df_plot.groupby("month").apply(
        lambda g: stats.spearmanr(g["x"], g["y"]).statistic
    ).rename("IC")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig2 = px.bar(x=month_names, y=monthly_ic.values,
                  color=monthly_ic.values,
                  color_continuous_scale="RdBu", range_color=[-0.2, 0.2],
                  labels={"x": "Month", "y": "Spearman IC"},
                  title="IC by calendar month — is the signal seasonal?")
    fig2.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Strong IC in summer/fall = drought signal peaks when snowmelt ends and "
               "reservoir depletion matters most.")


def _model_results():
    st.header("Model Results")

    if not MODEL_METRICS_FILE.exists():
        st.warning("Run `python scripts/main.py` to generate results.")
        return

    df = pd.read_csv(MODEL_METRICS_FILE)
    num_cols = ["ic", "hit_rate", "sharpe", "rmse", "mae", "r2"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            df[["model_name"] + num_cols]
            .style.format({c: "{:.4f}" for c in num_cols})
            .highlight_max(subset=["ic", "hit_rate", "sharpe", "r2"], color="#c8f5d0")
            .highlight_min(subset=["rmse", "mae"], color="#c8f5d0"),
            use_container_width=True,
        )
    with col2:
        best = df.loc[df["ic"].idxmax(), "model_name"]
        best_ic = df["ic"].max()
        best_hr = df.loc[df["ic"].idxmax(), "hit_rate"]
        best_sh = df.loc[df["ic"].idxmax(), "sharpe"]
        st.metric("Best model", best)
        st.metric("IC (test)", f"{best_ic:+.4f}")
        st.metric("Hit rate (test)", f"{best_hr:.1%}")
        st.metric("Signal Sharpe", f"{best_sh:+.3f}")

    fig = px.bar(df, x="model_name", y="ic", color="model_name",
                 text_auto=".4f",
                 title="Information Coefficient — Test set (2019–2025)",
                 labels={"model_name": "Model", "ic": "IC (Spearman)"})
    fig.add_hline(y=0.10, line_dash="dash", line_color="green",
                  annotation_text="IC=0.10 practical threshold")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _backtest(X_test, y_test):
    st.header("Signal Backtest")
    st.markdown(
        "Long IDA / short XLU when predicted excess return > 0; "
        "inverse when < 0. Daily rebalancing, **no transaction costs** (upper bound)."
    )

    available = {k: v["name"] for k, v in MODELS.items()
                 if Path(v["path"]).exists()}
    if not available:
        st.error("Train models first: `python scripts/train.py`")
        return

    key = st.selectbox("Model", list(available.keys()),
                       format_func=lambda k: available[k])
    model = load_model(key)
    if model is None:
        return

    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    signal = np.sign(y_pred)
    ls_ret = signal * y_test

    cum_signal = (1 + ls_ret).cumprod()
    cum_buynhold = (1 + y_test).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_signal.index, y=cum_signal,
                             name="L/S Signal", line=dict(color=HYDRO_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=cum_buynhold.index, y=cum_buynhold,
                             name="Buy & Hold IDA excess", line=dict(color=NEUTRAL, width=1.5,
                             dash="dot")))
    fig.add_hline(y=1, line_color="grey", line_dash="dash")
    fig.update_layout(title=f"Cumulative L/S return — {available[key]} (test set 2019–2025)",
                      yaxis_title="Cumulative return (base=1)",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    total = float(cum_signal.iloc[-1] - 1)
    ann_ret = float((cum_signal.iloc[-1]) ** (252 / len(cum_signal)) - 1)
    vol = float(ls_ret.std() * np.sqrt(252))
    sharpe = ann_ret / vol if vol > 0 else 0
    col1.metric("Total return", f"{total:+.1%}")
    col2.metric("Ann. return", f"{ann_ret:+.1%}")
    col3.metric("Ann. volatility", f"{vol:.1%}")
    col4.metric("Sharpe ratio", f"{sharpe:+.2f}")


def _feature_importance():
    st.header("Feature Importance")

    rf = load_model("random_forest")
    xgb = load_model("xgboost")
    if rf is None or xgb is None:
        st.error("Train models first.")
        return

    X_train, X_test, _, _ = load_dataset_split()[:4]
    feat_names = X_train.columns.tolist()
    labels = {c: c.replace("_", " ").title() for c in feat_names}

    col1, col2 = st.columns(2)
    for ax_col, (name, model, color) in zip(
        [col1, col2],
        [("Random Forest", rf, RETURN_UP), ("XGBoost", xgb, HYDRO_COLOR)],
    ):
        imp = model.named_steps["reg"].feature_importances_
        order = np.argsort(imp)
        fig = go.Figure(go.Bar(
            x=imp[order],
            y=[labels[feat_names[i]] for i in order],
            orientation="h",
            marker_color=[color if imp[i] > np.median(imp) else NEUTRAL for i in order],
        ))
        fig.update_layout(title=name, height=500,
                          xaxis_title="Importance score")
        ax_col.plotly_chart(fig, use_container_width=True)

    st.caption(
        "If Snake River z-score and deficit dominate → the signal is "
        "concentrated in drought anomalies on IDA's primary water source. "
        "If seasonal features dominate → residual seasonality in the target."
    )


# ── Entry point ────────────────────────────────────────────────────────────────
def build_app() -> None:
    st.set_page_config(page_title="Hydro-Alpha", page_icon="💧", layout="wide")
    st.title("💧 Hydro-Alpha")
    st.caption(
        "USGS river streamflow → IDACORP (IDA) excess return prediction · "
        "Alternative data · Quantitative signal research"
    )

    flow, stocks = load_raw()
    X_train, X_test, y_train, y_test, X_all, y_all = get_dataset()

    tabs = st.tabs([
        "Overview", "Streamflow", "Signal Analysis",
        "Model Results", "Backtest", "Feature Importance",
    ])
    with tabs[0]: _overview()
    with tabs[1]: _streamflow(flow, stocks)
    with tabs[2]: _signal_analysis(X_all, y_all)
    with tabs[3]: _model_results()
    with tabs[4]: _backtest(X_test, y_test)
    with tabs[5]: _feature_importance()


if __name__ == "__main__":
    build_app()
