"""Fetch and consolidate all raw data for the Hydro-Alpha project.

Downloads:
  1. USGS daily streamflow for 4 Pacific NW river gauges (2000–present)
  2. IDA and XLU adjusted close prices via yfinance (2000–present)
  3. Consolidates ICE electricity prices (MID-C hub) from local Excel files

Outputs:
  data/raw/hydro/usgs_streamflow_daily.csv
  data/raw/hydro/stock_prices_daily.csv
  data/raw/hydro/ice_midc_daily.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HYDRO_DIR = DATA_DIR / "raw" / "hydro"
HYDRO_DIR.mkdir(parents=True, exist_ok=True)

# ── USGS gauges ───────────────────────────────────────────────────────────────
USGS_GAUGES = {
    "columbia":   "14105700",  # Columbia River at The Dalles, OR
    "snake":      "13334300",  # Snake River near Anatone, WA
    "willamette": "14211720",  # Willamette River at Portland, OR
    "deschutes":  "14092500",  # Deschutes River near Madras, OR
}

START_DATE = "2000-01-01"


def fetch_usgs_streamflow() -> pd.DataFrame:
    """Download daily mean streamflow from USGS NWIS for all gauges."""
    frames = []
    for river, site_id in USGS_GAUGES.items():
        print(f"  Fetching USGS streamflow: {river} (site {site_id})...")
        url = (
            "https://waterservices.usgs.gov/nwis/dv/"
            f"?format=json&sites={site_id}"
            f"&startDT={START_DATE}"
            "&parameterCd=00060"  # discharge in cfs
            "&statCd=00003"       # mean daily
            "&siteStatus=all"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        ts = data["value"]["timeSeries"]
        if not ts:
            print(f"    WARNING: no data for {river}")
            continue

        values = ts[0]["values"][0]["value"]
        records = [
            {"date": v["dateTime"][:10], f"discharge_cfs_{river}": float(v["value"])}
            for v in values
            if v["value"] != "-999999"
        ]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        frames.append(df)
        time.sleep(0.5)

    result = pd.concat(frames, axis=1).sort_index()
    print(f"  Streamflow: {len(result)} days, {result.columns.tolist()}")
    return result


def fetch_stock_prices() -> pd.DataFrame:
    """Download adjusted close prices for IDA and XLU via yfinance."""
    print("  Fetching stock prices: IDA, XLU...")
    tickers = ["IDA", "XLU"]
    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
    prices = data["Close"][tickers]
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices = prices.dropna(how="all")
    print(f"  Stocks: {len(prices)} trading days ({prices.index[0].date()} -> {prices.index[-1].date()})")
    return prices


def consolidate_ice_midc() -> pd.DataFrame:
    """Parse all ICE Excel files and extract MID-C hub weighted average prices."""
    frames = []

    # Historical file (2001–2013)
    hist_file = DATA_DIR / "ice_electric-historical" / "MID-C Hub.xls"
    if hist_file.exists():
        print("  Parsing ICE historical MID-C Hub...")
        df = pd.read_excel(hist_file)
        df = df.rename(columns={
            "Trade Date": "date",
            "Wtd Avg Price $/MWh": "midc_price",
            "Daily Volume MWh": "midc_volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df[["date", "midc_price", "midc_volume"]])

    # Annual files (2014–2025)
    annual_files = sorted(DATA_DIR.glob("ice_electric-*final*.xls*"))
    for f in annual_files:
        if "historical" in str(f):
            continue
        print(f"  Parsing {f.name}...")
        df = pd.read_excel(f)

        col_map = {}
        for c in df.columns:
            cl = c.lower().replace("\n", " ").strip()
            if "hub" in cl or "price hub" in cl:
                col_map[c] = "hub"
            elif "trade date" in cl:
                col_map[c] = "date"
            elif "wtd avg" in cl:
                col_map[c] = "midc_price"
            elif "daily volume" in cl:
                col_map[c] = "midc_volume"
        df = df.rename(columns=col_map)

        if "hub" not in df.columns:
            continue

        midc = df[df["hub"].str.contains("Mid C", case=False, na=False)].copy()
        if midc.empty:
            continue

        midc["date"] = pd.to_datetime(midc["date"])
        frames.append(midc[["date", "midc_price", "midc_volume"]].copy())

    if not frames:
        print("  WARNING: no ICE MID-C data found")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("date").drop_duplicates(subset="date", keep="last")
    result = result.set_index("date")

    # Forward-fill gaps (weekends/holidays) then keep only business days
    full_idx = pd.date_range(result.index.min(), result.index.max(), freq="D")
    result = result.reindex(full_idx)
    result.index.name = "date"
    result["midc_price"] = result["midc_price"].ffill(limit=5)
    result["midc_volume"] = result["midc_volume"].fillna(0)

    print(f"  ICE MID-C: {result['midc_price'].notna().sum()} days with prices "
          f"({result.index[0].date()} -> {result.index[-1].date()})")
    return result


# ── SNOTEL stations (Snake River / central Idaho headwaters) ──────────────────
SNOTEL_STATIONS = {
    "bogus_basin":     "550:ID:SNTL",
    "dollarhide":      "489:ID:SNTL",
    "banner_summit":   "312:ID:SNTL",
    "trinity_mtn":     "774:ID:SNTL",
}


def fetch_snotel_snowpack() -> pd.DataFrame:
    """Download daily Snow Water Equivalent from SNOTEL stations in Idaho."""
    print("  Fetching SNOTEL snowpack data...")
    frames = []
    for name, triplet in SNOTEL_STATIONS.items():
        print(f"    Station: {name} ({triplet})...")
        url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
        params = {
            "stationTriplets": triplet,
            "elements": "WTEQ",
            "beginDate": START_DATE,
            "endDate": "2026-12-31",
            "duration": "DAILY",
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if not data or not data[0].get("data"):
                print(f"      WARNING: no data for {name}")
                continue

            values = data[0]["data"][0]["values"]
            records = [
                {"date": v["date"], f"swe_{name}": v["value"]}
                for v in values
                if v.get("value") is not None and v["value"] >= 0
            ]
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            frames.append(df)
        except Exception as e:
            print(f"      ERROR: {e}")
        time.sleep(0.5)

    if not frames:
        print("  WARNING: no SNOTEL data retrieved")
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).sort_index()
    swe_cols = [c for c in result.columns if c.startswith("swe_")]
    result["swe_mean"] = result[swe_cols].mean(axis=1)
    result = result.ffill(limit=3)
    print(f"  SNOTEL: {result['swe_mean'].notna().sum()} days "
          f"({result.index[0].date()} -> {result.index[-1].date()})")
    return result


def fetch_natural_gas() -> pd.DataFrame:
    """Download Henry Hub natural gas futures prices via yfinance."""
    print("  Fetching Henry Hub natural gas (NG=F)...")
    data = yf.download("NG=F", start=START_DATE, auto_adjust=True, progress=False)
    prices = data["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna()
    prices.name = "gas_price"
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Reindex to daily and forward-fill (weekends/holidays)
    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    result = prices.reindex(full_idx).ffill(limit=5)
    result = result.to_frame()
    result.index.name = "date"

    print(f"  Gas: {result['gas_price'].notna().sum()} days "
          f"({result.index[0].date()} -> {result.index[-1].date()})")
    return result


def main() -> None:
    print("=" * 60)
    print("Hydro-Alpha Data Fetch")
    print("=" * 60)

    print("\n[1/5] USGS Streamflow")
    flow = fetch_usgs_streamflow()
    flow_path = HYDRO_DIR / "usgs_streamflow_daily.csv"
    flow.to_csv(flow_path)
    print(f"  -> Saved to {flow_path.relative_to(PROJECT_ROOT)}")

    print("\n[2/5] Stock Prices")
    stocks = fetch_stock_prices()
    stocks_path = HYDRO_DIR / "stock_prices_daily.csv"
    stocks.to_csv(stocks_path)
    print(f"  -> Saved to {stocks_path.relative_to(PROJECT_ROOT)}")

    print("\n[3/5] ICE Electricity (MID-C Hub)")
    ice = consolidate_ice_midc()
    if not ice.empty:
        ice_path = HYDRO_DIR / "ice_midc_daily.csv"
        ice.to_csv(ice_path)
        print(f"  -> Saved to {ice_path.relative_to(PROJECT_ROOT)}")

    print("\n[4/5] SNOTEL Snowpack (Idaho)")
    snotel = fetch_snotel_snowpack()
    if not snotel.empty:
        snotel_path = HYDRO_DIR / "snotel_swe_daily.csv"
        snotel.to_csv(snotel_path)
        print(f"  -> Saved to {snotel_path.relative_to(PROJECT_ROOT)}")

    print("\n[5/5] Henry Hub Natural Gas")
    gas = fetch_natural_gas()
    if not gas.empty:
        gas_path = HYDRO_DIR / "henry_hub_gas_daily.csv"
        gas.to_csv(gas_path)
        print(f"  -> Saved to {gas_path.relative_to(PROJECT_ROOT)}")

    print("\nDone. All data saved to data/raw/hydro/")


if __name__ == "__main__":
    main()
