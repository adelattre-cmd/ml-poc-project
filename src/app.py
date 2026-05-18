"""Streamlit app — Hydro-Alpha: USGS Streamflow → IDACORP Excess Return."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from sklearn.base import clone

from config import (
    DATA_DIR, MODEL_METRICS_FILE, MODELS, MODELS_DIR,
    TARGET_TICKER, BENCH_TICKER, FORWARD_DAYS, RESULTS_DIR,
)
from data import (
    build_features, build_target, load_dataset_split,
    FLOW_FILE, STOCKS_FILE, TRAIN_END, TEST_START, RIVERS,
)

HYDRO_COLOR  = "#1d6fa5"
RETURN_UP    = "#2a9d8f"
RETURN_DOWN  = "#e63946"
NEUTRAL      = "#adb5bd"
TEST_START_TS = pd.Timestamp(TEST_START)


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


@st.cache_data(show_spinner=False)
def load_run_report() -> dict:
    report_path = RESULTS_DIR / "run_report.json"
    if not report_path.exists():
        return {}

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


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
        test_start_dt = TEST_START_TS.to_pydatetime()
        fig.add_shape(
            type="line",
            x0=test_start_dt,
            x1=test_start_dt,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="grey", dash="dash"),
        )
        fig.add_annotation(
            x=test_start_dt,
            y=1,
            xref="x",
            yref="paper",
            text="Test start",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="grey"),
        )
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
                            "period": np.where(X_all.index < TEST_START_TS, "Train", "Test")})
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


def _compute_perf_stats(returns: pd.Series, holding_period: int) -> dict[str, float]:
    if returns.empty:
        return {
            "total": 0.0,
            "ann_ret": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }

    positions_per_year = 252 / holding_period
    cum = (1 + returns).cumprod()
    n = len(returns)
    total = float(cum.iloc[-1] - 1)
    ann_ret = float((cum.iloc[-1]) ** (positions_per_year / n) - 1) if n > 0 else 0.0
    ann_vol = float(returns.std() * np.sqrt(positions_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    drawdown = cum / cum.cummax() - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "total": total,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def _build_entry_predictions(
    base_model,
    retrain_mode: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    holding_period: int,
    min_hist: int,
    warmup_days: int,
    entry_offset: int,
) -> tuple[pd.Series, pd.Series, int]:
    mask_trading = y_test != y_test.shift(1)
    mask_trading.iloc[0] = True
    X_trading = X_test[mask_trading]
    y_trading = y_test[mask_trading]

    entry_idx = np.arange(entry_offset, len(y_trading), holding_period)
    if warmup_days > 0:
        entry_idx = entry_idx[entry_idx >= warmup_days]

    returns = y_trading.iloc[entry_idx]

    if retrain_mode.startswith("Static"):
        y_pred = pd.Series(base_model.predict(X_trading), index=X_trading.index)
        positions = y_pred.iloc[entry_idx]
        return positions.sort_index(), returns.loc[positions.index], 0

    X_full = pd.concat([X_train, X_test]).sort_index()
    y_full = pd.concat([y_train, y_test]).sort_index()

    mask_full_trading = y_full != y_full.shift(1)
    mask_full_trading.iloc[0] = True
    X_full_trading = X_full[mask_full_trading]
    y_full_trading = y_full[mask_full_trading]

    preds: list[float] = []
    pred_dates = []
    skipped = 0

    for i in entry_idx:
        pred_date = X_trading.index[i]
        pos_full = y_full_trading.index.get_indexer([pred_date])[0]
        train_end_pos = pos_full - FORWARD_DAYS
        if train_end_pos < min_hist:
            skipped += 1
            continue

        X_fit = X_full_trading.iloc[: train_end_pos + 1]
        y_fit = y_full_trading.iloc[: train_end_pos + 1]
        if len(X_fit) == 0 or len(X_fit) != len(y_fit):
            skipped += 1
            continue

        model_step = clone(base_model)
        model_step.fit(X_fit, y_fit)
        preds.append(float(model_step.predict(X_trading.iloc[[i]])[0]))
        pred_dates.append(pred_date)

    if not preds:
        return pd.Series(dtype=float), pd.Series(dtype=float), skipped

    positions = pd.Series(preds, index=pred_dates).sort_index()
    returns = returns.loc[positions.index]
    return positions, returns, skipped


def _apply_execution_layer(
    preds: pd.Series,
    realized: pd.Series,
    entry_threshold: float,
    use_two_stage: bool,
    confidence_ratio: float,
    sizing_mode: str,
    max_leverage: float,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if preds.empty or realized.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    abs_pred = preds.abs()
    score_ref = abs_pred.expanding(min_periods=10).median().shift(1)
    score_ref = score_ref.fillna(abs_pred.median() if abs_pred.median() > 0 else 1e-6)
    confidence_score = abs_pred / (score_ref + 1e-9)

    raw_signal = np.sign(preds) * (abs_pred >= entry_threshold).astype(float)
    if use_two_stage:
        raw_signal = raw_signal * (confidence_score >= confidence_ratio).astype(float)

    if sizing_mode == "Dynamic (scaled by confidence)":
        size = np.clip(confidence_score / max(confidence_ratio, 1e-6), 0.0, max_leverage)
        position = raw_signal * size
    else:
        position = raw_signal

    turnover = position.diff().abs().fillna(position.abs())
    gross = position * realized
    costs = turnover * (cost_bps / 10000)
    net = gross - costs
    return net, gross, turnover


def _bootstrap_sharpe_ci(returns: pd.Series, holding_period: int, n_boot: int = 500) -> tuple[float, float]:
    if returns.empty:
        return 0.0, 0.0
    arr = returns.to_numpy(dtype=float)
    if len(arr) < 5:
        return 0.0, 0.0
    ppy = 252 / holding_period
    samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        s = rng.choice(arr, size=len(arr), replace=True)
        vol = float(np.std(s) * np.sqrt(ppy))
        ann = float((np.prod(1 + s) ** (ppy / len(s))) - 1)
        samples.append(ann / vol if vol > 0 else 0.0)
    return float(np.percentile(samples, 5)), float(np.percentile(samples, 95))


def _backtest(X_train, X_test, y_train, y_test):
    st.header("Signal Backtest")
    st.markdown(
        "Execution layer includes thresholding, optional 2-stage filtering, dynamic sizing, "
        "transaction costs, and walk-forward retraining."
    )

    available = {k: v["name"] for k, v in MODELS.items()
                 if Path(v["path"]).exists()}
    if not available:
        st.error("Train models first: `python scripts/train.py`")
        return

    setup_mode = st.radio(
        "Setup mode",
        ["Guided (recommended)", "Expert (manual)"],
        horizontal=True,
    )
    retrain_mode = st.selectbox(
        "Backtest mode",
        ["Walk-forward (retrain each trade)", "Static model (no retrain)"],
        index=0,
    )

    available_keys = list(available.keys())
    default_model_key = "pca_ridge" if "pca_ridge" in available_keys else available_keys[0]
    key = st.selectbox(
        "Model",
        available_keys,
        index=available_keys.index(default_model_key),
        format_func=lambda k: available[k],
    )

    if setup_mode.startswith("Guided"):
        profile = st.selectbox(
            "Risk profile",
            ["Recommended", "Balanced", "Conservative", "Aggressive"],
            index=0,
        )
        presets = {
            "Recommended": {
                "model_key": "pca_ridge",
                "holding": 50,
                "warmup": 40,
                "offset": 0,
                "threshold": 0.005,
                "use_two_stage": True,
                "confidence": 1.0,
                "sizing": "Dynamic (scaled by confidence)",
                "leverage": 1.0,
                "cost_bps": 5.0,
            },
            "Conservative": {
                "model_key": "ridge",
                "holding": 30,
                "warmup": 60,
                "offset": 0,
                "threshold": 0.008,
                "use_two_stage": True,
                "confidence": 1.2,
                "sizing": "Binary (-1/0/+1)",
                "leverage": 1.0,
                "cost_bps": 7.0,
            },
            "Balanced": {
                "model_key": "pca_ridge",
                "holding": 25,
                "warmup": 40,
                "offset": 0,
                "threshold": 0.005,
                "use_two_stage": True,
                "confidence": 1.0,
                "sizing": "Dynamic (scaled by confidence)",
                "leverage": 1.0,
                "cost_bps": 5.0,
            },
            "Aggressive": {
                "model_key": "ridge",
                "holding": 20,
                "warmup": 20,
                "offset": 0,
                "threshold": 0.003,
                "use_two_stage": False,
                "confidence": 0.9,
                "sizing": "Dynamic (scaled by confidence)",
                "leverage": 1.3,
                "cost_bps": 3.0,
            },
        }
        cfg = presets[profile]
        if cfg["model_key"] in available:
            key = cfg["model_key"]
        holding_period = cfg["holding"]
        warmup_days = cfg["warmup"]
        entry_offset = cfg["offset"]
        entry_threshold = cfg["threshold"]
        use_two_stage = cfg["use_two_stage"]
        confidence_ratio = cfg["confidence"]
        sizing_mode = cfg["sizing"]
        max_leverage = cfg["leverage"]
        cost_bps = cfg["cost_bps"]
        st.caption(
            f"Preset `{profile}` applied: hold={holding_period}, threshold={entry_threshold:.3f}, "
            f"confidence={confidence_ratio:.1f}, cost={cost_bps:.1f} bps."
        )
    else:
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            holding_period = st.slider("Holding period (trading days)", 5, 60, FORWARD_DAYS, 5)
            warmup_days = st.slider("Warm-up before first trade (trading days)", 0, 252, 40, 5)
            entry_offset = st.slider("Entry offset (robustness)", 0, max(0, holding_period - 1), 0, 1)
        with col_cfg2:
            entry_threshold = st.slider("Entry threshold on |prediction|", 0.0, 0.05, 0.005, 0.001)
            use_two_stage = st.checkbox("Use 2-stage filter (confidence gate)", value=True)
            confidence_ratio = st.slider("Confidence ratio threshold", 0.5, 2.0, 1.0, 0.1)
        with col_cfg3:
            sizing_mode = st.selectbox("Position sizing", ["Binary (-1/0/+1)", "Dynamic (scaled by confidence)"])
            max_leverage = st.slider("Max position leverage (dynamic mode)", 0.5, 2.0, 1.0, 0.1)
            cost_bps = st.slider("Transaction cost (bps per turnover unit)", 0.0, 50.0, 5.0, 0.5)

    model = load_model(key)
    if model is None:
        return

    min_hist = max(120, FORWARD_DAYS * 6)
    if st.button("Auto-select best model + holding (fast)"):
        scan_models = [mk for mk in available_keys if mk != "xgboost"]
        scan_holdings = [15, 20, 25, 30, 35, 40, 45, 50]
        scan_offsets = [0, 1, 2]
        best = None
        total_jobs = sum(len([o for o in scan_offsets if o < hp]) for hp in scan_holdings) * len(scan_models)
        done_jobs = 0
        time_budget_s = 25.0
        started = time.time()
        progress = st.progress(0, text="Fast scan in progress...")
        stopped_early = False
        with st.spinner("Searching robust settings (fast scan)..."):
            for mk in scan_models:
                m = load_model(mk)
                if m is None:
                    continue
                for hp in scan_holdings:
                    sharpes = []
                    for off in [o for o in scan_offsets if o < hp]:
                        if time.time() - started > time_budget_s:
                            stopped_early = True
                            break
                        p, r, _ = _build_entry_predictions(
                            base_model=m,
                            retrain_mode=retrain_mode,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            holding_period=hp,
                            min_hist=min_hist,
                            warmup_days=warmup_days,
                            entry_offset=off,
                        )
                        if p.empty or r.empty:
                            continue
                        net, _, _ = _apply_execution_layer(
                            preds=p,
                            realized=r,
                            entry_threshold=entry_threshold,
                            use_two_stage=use_two_stage,
                            confidence_ratio=confidence_ratio,
                            sizing_mode=sizing_mode,
                            max_leverage=max_leverage,
                            cost_bps=cost_bps,
                        )
                        if net.empty:
                            continue
                        sharpes.append(_compute_perf_stats(net, hp)["sharpe"])
                        done_jobs += 1
                        if total_jobs > 0:
                            pct = min(100, int((done_jobs / total_jobs) * 100))
                            progress.progress(pct, text=f"Fast scan in progress... {pct}%")
                    if stopped_early:
                        break
                    if not sharpes:
                        continue
                    score = float(np.median(sharpes))
                    if best is None or score > best["score"]:
                        best = {"model": mk, "holding": hp, "score": score}
                if stopped_early:
                    break
        progress.empty()
        if best is not None:
            key = best["model"]
            holding_period = int(best["holding"])
            model = load_model(key)
            msg = (
                f"Recommended settings: model={available[key]}, holding={holding_period}, "
                f"median Sharpe={best['score']:+.3f}."
            )
            if stopped_early:
                st.warning(msg + " Returned best partial result (time budget reached).")
            else:
                st.success(msg)
        else:
            st.warning("No valid combination found in fast scan under current filters.")

    preds, returns, skipped = _build_entry_predictions(
        base_model=model,
        retrain_mode=retrain_mode,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        holding_period=holding_period,
        min_hist=min_hist,
        warmup_days=warmup_days,
        entry_offset=entry_offset,
    )
    if preds.empty or returns.empty:
        st.error("Backtest could not run: no valid entry points under current settings.")
        return

    net_ret, gross_ret, turnover = _apply_execution_layer(
        preds=preds,
        realized=returns,
        entry_threshold=entry_threshold,
        use_two_stage=use_two_stage,
        confidence_ratio=confidence_ratio,
        sizing_mode=sizing_mode,
        max_leverage=max_leverage,
        cost_bps=cost_bps,
    )

    if net_ret.empty:
        st.error("No trades after filters. Lower threshold/confidence gate.")
        return

    cum_signal = (1 + net_ret).cumprod()
    cum_buynhold = (1 + returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_signal.index, y=cum_signal,
                             name="L/S Signal", line=dict(color=HYDRO_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=cum_buynhold.index, y=cum_buynhold,
                             name="Buy & Hold IDA excess", line=dict(color=NEUTRAL, width=1.5,
                             dash="dot")))
    fig.add_hline(y=1, line_color="grey", line_dash="dash")
    fig.update_layout(
        title=f"Cumulative L/S return — {available[key]} (test 2019–2025, hold={holding_period}d)",
        yaxis_title="Cumulative return (base=1)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    stats_net = _compute_perf_stats(net_ret, holding_period)
    ci_low, ci_high = _bootstrap_sharpe_ci(net_ret, holding_period, n_boot=300)
    turnover_annual = float(turnover.mean() * (252 / holding_period)) if not turnover.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total return (net)", f"{stats_net['total']:+.1%}")
    col2.metric("Ann. return (net)", f"{stats_net['ann_ret']:+.1%}")
    col3.metric("Sharpe (net)", f"{stats_net['sharpe']:+.2f}")
    col4.metric("Max drawdown", f"{stats_net['max_drawdown']:.1%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Calmar ratio", f"{stats_net['calmar']:+.2f}")
    col6.metric("Ann. volatility", f"{stats_net['ann_vol']:.1%}")
    col7.metric("Turnover/year", f"{turnover_annual:.2f}")
    col8.metric("Sharpe 90% CI", f"[{ci_low:+.2f}, {ci_high:+.2f}]")

    # Baseline naive strategy: sign of snake_zscore (if available), else sign of target lag-1.
    mask_trading = y_test != y_test.shift(1)
    mask_trading.iloc[0] = True
    X_trading = X_test[mask_trading]
    y_trading = y_test[mask_trading]
    entry_idx = np.arange(entry_offset, len(y_trading), holding_period)
    if warmup_days > 0:
        entry_idx = entry_idx[entry_idx >= warmup_days]
    base_returns = y_trading.iloc[entry_idx]
    if "snake_zscore" in X_trading.columns:
        base_signal = np.sign(X_trading["snake_zscore"].iloc[entry_idx]).astype(float)
    else:
        base_signal = np.sign(y_trading.shift(1).iloc[entry_idx]).fillna(0.0).astype(float)
    baseline_ret = base_signal * base_returns
    baseline_ret = baseline_ret.loc[net_ret.index.intersection(baseline_ret.index)]
    net_aligned = net_ret.loc[baseline_ret.index]
    baseline_stats = _compute_perf_stats(baseline_ret, holding_period)

    st.markdown("**Robustness Checks**")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Baseline Sharpe", f"{baseline_stats['sharpe']:+.2f}")
        st.metric("Baseline total return", f"{baseline_stats['total']:+.1%}")
    with r2:
        diff = stats_net["sharpe"] - baseline_stats["sharpe"]
        st.metric("Sharpe uplift vs baseline", f"{diff:+.2f}")
        outperf = float((net_aligned - baseline_ret).sum())
        st.metric("Cumulative excess vs baseline", f"{outperf:+.1%}")

    # Sub-period stability table
    seg = pd.DataFrame({"ret": net_ret})
    seg["period"] = pd.cut(
        seg.index.year,
        bins=[2018, 2020, 2022, 2024, 2027],
        labels=["2019-2020", "2021-2022", "2023-2024", "2025-2026"],
        right=True,
    )
    sub_rows = []
    for label, grp in seg.groupby("period", observed=True):
        if grp.empty:
            continue
        s = _compute_perf_stats(grp["ret"], holding_period)
        sub_rows.append(
            {
                "period": str(label),
                "trades": int(len(grp)),
                "total": s["total"],
                "ann_ret": s["ann_ret"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_drawdown"],
            }
        )
    if sub_rows:
        st.dataframe(
            pd.DataFrame(sub_rows).style.format(
                {"total": "{:+.2%}", "ann_ret": "{:+.2%}", "sharpe": "{:+.2f}", "max_dd": "{:.2%}"}
            ),
            use_container_width=True,
        )

    st.markdown("**Robust Model/Holding Scan (multi-offset median Sharpe)**")
    if st.button("Run robust scan"):
        scan_models = [
            mk for mk, cfg in MODELS.items() if Path(cfg["path"]).exists() and mk != "xgboost"
        ]
        scan_holdings = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        offsets = [0, 1, 2]
        results = []
        total_jobs = sum(len([o for o in offsets if o < hp]) for hp in scan_holdings) * len(scan_models)
        done_jobs = 0
        time_budget_s = 30.0
        started = time.time()
        progress = st.progress(0, text="Robust scan in progress...")
        stopped_early = False
        with st.spinner("Running scan..."):
            for mk in scan_models:
                m = load_model(mk)
                if m is None:
                    continue
                for hp in scan_holdings:
                    sharpes = []
                    for off in [o for o in offsets if o < hp]:
                        if time.time() - started > time_budget_s:
                            stopped_early = True
                            break
                        p, r, _ = _build_entry_predictions(
                            base_model=m,
                            retrain_mode=retrain_mode,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            holding_period=hp,
                            min_hist=min_hist,
                            warmup_days=warmup_days,
                            entry_offset=off,
                        )
                        if p.empty or r.empty:
                            continue
                        net, _, _ = _apply_execution_layer(
                            preds=p,
                            realized=r,
                            entry_threshold=entry_threshold,
                            use_two_stage=use_two_stage,
                            confidence_ratio=confidence_ratio,
                            sizing_mode=sizing_mode,
                            max_leverage=max_leverage,
                            cost_bps=cost_bps,
                        )
                        if net.empty:
                            continue
                        sharpes.append(_compute_perf_stats(net, hp)["sharpe"])
                        done_jobs += 1
                        if total_jobs > 0:
                            pct = min(100, int((done_jobs / total_jobs) * 100))
                            progress.progress(pct, text=f"Robust scan in progress... {pct}%")
                    if stopped_early:
                        break
                    if sharpes:
                        results.append(
                            {
                                "model": mk,
                                "holding": hp,
                                "median_sharpe": float(np.median(sharpes)),
                                "min_sharpe": float(np.min(sharpes)),
                                "max_sharpe": float(np.max(sharpes)),
                                "n_offsets": int(len(sharpes)),
                            }
                        )
                if stopped_early:
                    break
        progress.empty()
        if results:
            scan_df = pd.DataFrame(results).sort_values("median_sharpe", ascending=False)
            st.dataframe(
                scan_df.style.format(
                    {"median_sharpe": "{:+.3f}", "min_sharpe": "{:+.3f}", "max_sharpe": "{:+.3f}"}
                ),
                use_container_width=True,
            )
            best = scan_df.iloc[0]
            st.success(
                f"Best robust combo: {best['model']} with holding={int(best['holding'])} "
                f"(median Sharpe {best['median_sharpe']:+.3f})."
            )
            if stopped_early:
                st.info("Robust scan stopped at time budget; showing best partial ranking.")

    st.caption(
        f"{len(net_ret)} non-overlapping entries | mode: {retrain_mode} | "
        f"threshold={entry_threshold:.3f} | cost={cost_bps:.1f} bps"
    )
    if retrain_mode.startswith("Walk-forward"):
        st.caption(
            f"Walk-forward uses train+elapsed test data only. "
            f"Skipped entries due to insufficient history: {skipped}."
        )


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


def _executive_demo(X_train, X_test, y_train, y_test, X_all, y_all):
    st.header("Demo Mode: 5-Minute Presentation")
    st.markdown(
        "Use this guided script to present the project end-to-end with a clear story "
        "for non-technical or mixed audiences."
    )

    step = st.radio(
        "Presentation step",
        [
            "1) Business problem",
            "2) Data and signal intuition",
            "3) Feature design logic",
            "4) Model comparison",
            "5) Backtest and decision",
        ],
        horizontal=True,
    )

    if step == "1) Business problem":
        st.subheader("Why this matters")
        st.write(
            f"We predict whether `{TARGET_TICKER}` will outperform `{BENCH_TICKER}` "
            f"over the next {FORWARD_DAYS} trading days."
        )
        st.write(
            "Hypothesis: river-flow anomalies impact hydropower costs, then earnings, "
            "then relative stock performance."
        )
        st.info(
            "Key message: this is an alternative-data signal built from public USGS data."
        )

    elif step == "2) Data and signal intuition":
        st.subheader("What data we use")
        st.write(
            "Inputs: daily USGS streamflow for 4 river gauges + daily adjusted prices "
            "for IDA and XLU."
        )
        st.write(
            "Target: forward excess return (IDA - XLU). Chronological split avoids leakage."
        )

        probe_feature = "snake_zscore" if "snake_zscore" in X_all.columns else X_all.columns[0]
        tmp = pd.DataFrame(
            {
                "feature": X_all[probe_feature],
                "target": y_all,
            }
        ).dropna()
        quick_ic = stats.spearmanr(tmp["feature"], tmp["target"]).statistic
        st.metric("Example signal IC", f"{quick_ic:+.4f}", help=f"Feature: {probe_feature}")
        st.caption("IC > 0 suggests useful rank-ordering information in the signal.")

    elif step == "3) Feature design logic":
        st.subheader("How raw flow becomes model-ready features")
        st.markdown(
            "- **Z-score** removes seasonality and keeps anomalies only.\n"
            "- **Percentile** captures rarity (historically dry/wet conditions).\n"
            "- **Trend + deficit** capture persistence and regime pressure.\n"
            "- **Stock relative momentum** adds market context."
        )
        st.write(f"Feature count: {X_train.shape[1]}")
        st.code("\n".join(list(X_train.columns[:12]) + (["..."] if X_train.shape[1] > 12 else [])))

    elif step == "4) Model comparison":
        st.subheader("Which model is best on unseen data?")
        if not MODEL_METRICS_FILE.exists():
            st.warning("Run `python scripts/main.py` first to generate model metrics.")
            return
        df = pd.read_csv(MODEL_METRICS_FILE).sort_values("ic", ascending=False)
        st.dataframe(df, use_container_width=True)
        best = df.iloc[0]
        st.success(
            f"Best model by IC: {best['model_name']} (IC={best['ic']:+.4f}, "
            f"Hit Rate={best['hit_rate']:.1%})."
        )

    elif step == "5) Backtest and decision":
        st.subheader("So what would we do in production?")
        st.markdown(
            "- Use model sign for directional L/S allocation versus benchmark.\n"
            "- Retrain on rolling windows and monitor IC drift by month/regime.\n"
            "- Add costs, slippage and risk limits before live deployment."
        )
        st.write(
            f"Current dataset: {len(X_train):,} train rows, {len(X_test):,} test rows, "
            f"horizon {FORWARD_DAYS} trading days."
        )
        st.info("Decision framing: keep as research signal, then move to robust validation.")


def _full_process_tab(X_train, X_test, y_train, y_test):
    st.header("Project Walkthrough: End-to-End Process")
    st.markdown(
        "This tab explains each technical step, the reasoning behind it, and the artifacts "
        "you can inspect to present the project clearly."
    )

    report = load_run_report()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train rows", f"{len(X_train):,}")
        st.metric("Test rows", f"{len(X_test):,}")
    with col2:
        st.metric("Features", X_train.shape[1])
        st.metric("Target horizon", f"{FORWARD_DAYS}d")
    with col3:
        st.metric("Train period end", str(X_train.index.max().date()))
        st.metric("Test period start", str(X_test.index.min().date()))

    with st.expander("1) Problem framing and objective", expanded=True):
        st.write(
            f"Objective: predict the forward excess return of `{TARGET_TICKER}` versus "
            f"`{BENCH_TICKER}` using USGS streamflow anomalies."
        )
        st.write(
            "Rationale: river flow anomalies can impact hydropower generation costs, then "
            "margins, then relative equity performance."
        )

    with st.expander("2) Data sources and assumptions"):
        st.write("Files used:")
        st.code(
            "\n".join(
                [
                    f"- {FLOW_FILE}",
                    f"- {STOCKS_FILE}",
                ]
            )
        )
        st.write(
            "Main assumptions: chronological split (no leakage), no target shuffling, "
            "and forward returns computed on trading days."
        )

    with st.expander("3) Feature engineering choices"):
        st.write("Per river: z-score, percentile, trend, cumulative deficit.")
        st.write("Cross-sectional additions: seasonal encoding + stock relative momentum.")
        st.write(
            "Reasoning: isolate anomalies from seasonality, capture persistence of drought, "
            "and keep a market-relative stock context."
        )

    with st.expander("4) Modeling and evaluation strategy"):
        st.write("Registered models and rationale:")
        for key, cfg in MODELS.items():
            st.markdown(f"- **{cfg.get('name', key)}**: {cfg.get('description', '').strip()}")
        st.write(
            "Evaluation metrics combine signal quality (IC, hit rate, Sharpe) and "
            "regression quality (RMSE, MAE, R2)."
        )

    with st.expander("5) Backtest logic and limitations"):
        st.write(
            "Backtest rule: long when predicted excess return > 0, short otherwise; "
            "non-overlapping positions over holding windows."
        )
        st.write(
            "Limitations to mention in presentation: no transaction costs, no slippage, "
            "and no capacity constraints."
        )

    with st.expander("6) Reproducible run artifacts"):
        if MODEL_METRICS_FILE.exists():
            st.success(f"Metrics file available: {MODEL_METRICS_FILE}")
        else:
            st.warning("Metrics file missing. Run `python scripts/main.py`.")

        if report:
            st.write("Last orchestration report (`results/run_report.json`):")
            st.json(report)
        else:
            st.info(
                "Run report not found yet. It will be generated by `python scripts/main.py`."
            )

    with st.expander("7) Reflection and next improvements"):
        st.markdown(
            "- Validate robustness with rolling/expanding walk-forward retraining.\n"
            "- Add realistic costs and turnover controls in backtests.\n"
            "- Stress-test signal under drought/non-drought macro regimes.\n"
            "- Add CI tests for data contracts and model artifact checks."
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
        "Overview", "Demo Mode", "Process Walkthrough", "Streamflow", "Signal Analysis",
        "Model Results", "Backtest", "Feature Importance",
    ])
    with tabs[0]: _overview()
    with tabs[1]: _executive_demo(X_train, X_test, y_train, y_test, X_all, y_all)
    with tabs[2]: _full_process_tab(X_train, X_test, y_train, y_test)
    with tabs[3]: _streamflow(flow, stocks)
    with tabs[4]: _signal_analysis(X_all, y_all)
    with tabs[5]: _model_results()
    with tabs[6]: _backtest(X_train, X_test, y_train, y_test)
    with tabs[7]: _feature_importance()


if __name__ == "__main__":
    build_app()
