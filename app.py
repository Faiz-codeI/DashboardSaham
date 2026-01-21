import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="IDX Stock Health Dashboard (Teknikal + Fundamental)", layout="wide")

# =========================
# CONFIG
# =========================
DEFAULT_FUNDAMENTAL_CSV = r"C:\Users\Faiz\Documents\Magang\SahamLQ45\fundamental_scored_lq45_2024.csv"

LQ45 = [
    "AADI.JK","ACES.JK","ADMR.JK","ADRO.JK","AKRA.JK",
    "AMMN.JK","AMRT.JK","ANTM.JK","ASII.JK","BBCA.JK",
    "BBNI.JK","BBRI.JK","BBTN.JK","BMRI.JK","BRPT.JK",
    "BUMI.JK","CPIN.JK","CTRA.JK","DSSA.JK","EMTK.JK",
    "EXCL.JK","GOTO.JK","HEAL.JK","ICBP.JK","INCO.JK",
    "INDF.JK","INKP.JK","ISAT.JK","ITMG.JK","JPFA.JK",
    "KLBF.JK","MAPI.JK","MBMA.JK","MDKA.JK","MEDC.JK",
    "NCKL.JK","PGAS.JK","PGEO.JK","PTBA.JK","SCMA.JK",
    "SMGR.JK","TLKM.JK","TOWR.JK","UNTR.JK","UNVR.JK"
]

# =========================
# Utils (indikator)
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    dd = (close / peak) - 1
    return float(dd.min())  # negatif

def clamp(x, low=0, high=100) -> float:
    return max(low, min(high, float(x)))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def to_base_ticker(x: str) -> str:
    """
    UNVR.JK -> UNVR
    BBCA.JK -> BBCA
    """
    if x is None:
        return ""
    s = str(x).strip()
    return s.replace(".JK", "")

def infer_ticker_column(df: pd.DataFrame) -> str | None:
    candidates = ["ticker", "symbol", "kode", "emiten", "stock", "saham", "code"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    # fallback: kolom object yang mayoritas mirip kode emiten
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(80)
            if len(sample) and (sample.str.match(r"^[A-Z]{3,5}(\.JK)?$").mean() > 0.6):
                return c
    return None

def infer_score_column(df: pd.DataFrame) -> str | None:
    candidates = ["score", "total_score", "scored", "nilai", "final_score", "fundamental_score", "fund_score"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    # fallback: kolom numerik yang namanya mengandung score
    for c in df.columns:
        if "score" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback: ambil kolom numerik pertama
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[0] if num_cols else None

# =========================
# Load data
# =========================
@st.cache_data(ttl=3600)
def load_prices(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = yf.download(t, start=start, group_by="column", progress=False)
        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()  # Date jadi kolom

        # indikator
        df["ret_1d"] = df["Close"].pct_change()
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma50"] = df["Close"].rolling(50).mean()
        df["ma200"] = df["Close"].rolling(200).mean()
        df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        df["vol_avg20"] = df["Volume"].rolling(20).mean()
        df["rsi14"] = rsi(df["Close"], 14)

        out[t] = df

    return out

def health_from_df(df: pd.DataFrame) -> dict:
    d = df.dropna()
    if len(d) < 220:
        return {
            "health": np.nan, "trend": np.nan, "risk": np.nan, "liq": np.nan,
            "mdd": np.nan, "close": np.nan, "rsi": np.nan, "vol20": np.nan, "vol_ratio": np.nan
        }

    latest = d.iloc[-1]

    close = float(latest["Close"])
    ma50 = float(latest["ma50"])
    ma200 = float(latest["ma200"])
    vol20 = float(latest["vol_20d"])
    rsi14 = float(latest["rsi14"])
    vol_ratio = float(latest["Volume"] / latest["vol_avg20"])
    mdd = max_drawdown(d["Close"])

    # Trend (0/50/100)
    trend_score = (close > ma50) * 50 + (close > ma200) * 50

    # Risk (0..100) - makin tinggi makin aman
    risk_score = clamp(100 - (vol20 * 200) - (abs(mdd) * 200))

    # Liquidity (0..100) - makin tinggi makin likuid
    liq_score = clamp(60 + (vol_ratio - 1) * 20)

    # Total Health (0..100) - bobot (catatan: total bobot sekarang 0.80)
    health = clamp(0.40 * trend_score + 0.25 * risk_score + 0.15 * liq_score)

    return {
        "health": float(health),
        "trend": float(trend_score),
        "risk": float(risk_score),
        "liq": float(liq_score),
        "mdd": float(mdd),
        "close": float(close),
        "rsi": float(rsi14),
        "vol20": float(vol20),
        "vol_ratio": float(vol_ratio),
    }

@st.cache_data(show_spinner=False)
def load_fundamental_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path)
    return normalize_columns(df)

# =========================
# UI
# =========================
st.title("📊 IDX Stock Dashboard (LQ45) — Teknikal + Fundamental")

with st.sidebar:
    st.header("Pengaturan Teknikal")
    start = st.date_input("Mulai data harga dari", value=pd.to_datetime("2020-01-01"))

    picked = st.multiselect(
        "Pilih saham LQ45",
        options=LQ45,
        default=st.session_state.get("picked", ["UNVR.JK"])
    )
    st.session_state["picked"] = picked

    st.divider()
    st.header("Pengaturan Fundamental")
    fund_path = st.text_input("Path CSV Fundamental (scored)", value=DEFAULT_FUNDAMENTAL_CSV)

    st.divider()
    st.header("Pengaturan Gabungan")
    w_tech = st.slider("Bobot Teknikal", 0.0, 1.0, 0.5, 0.05)
    w_fund = st.slider("Bobot Fundamental", 0.0, 1.0, 0.5, 0.05)

    if not picked:
        st.stop()

tab_tech, tab_fund, tab_combo = st.tabs(["📈 Teknikal", "🧾 Fundamental", "🧩 Gabungan"])

# =========================
# TAB 1: TECHNICAL
# =========================
with tab_tech:
    data = load_prices(picked, start=str(start))

    if not data:
        st.warning("Tidak ada data yang berhasil diambil. Coba cek ticker atau koneksi.")
        st.stop()

    # Ringkasan skor teknikal
    rows = []
    for t, df in data.items():
        s = health_from_df(df)
        rows.append({
            "ticker": t,
            "ticker_base": to_base_ticker(t),
            "health_tech": s["health"],
            "trend": s["trend"],
            "risk": s["risk"],
            "liquidity": s["liq"],
            "close": s["close"],
            "rsi14": s["rsi"],
            "vol_20d": s["vol20"],
            "max_drawdown": s["mdd"],
            "vol_ratio": s["vol_ratio"],
        })

    summary = pd.DataFrame(rows).dropna(subset=["health_tech"]).sort_values("health_tech", ascending=False)

    if summary.empty:
        st.warning("Skor belum bisa dihitung (data kurang panjang atau banyak NaN).")
        st.stop()

    st.subheader("🏁 Ranking Kesehatan Saham (teknikal)")
    st.dataframe(
        summary.style.format({
            "health_tech": "{:.1f}",
            "trend": "{:.0f}",
            "risk": "{:.1f}",
            "liquidity": "{:.1f}",
            "close": "{:,.0f}",
            "rsi14": "{:.1f}",
            "vol_20d": "{:.2f}",
            "max_drawdown": "{:.1%}",
            "vol_ratio": "{:.2f}",
        }),
        use_container_width=True
    )

    st.caption("Skor teknikal: Trend (MA50/MA200), Risk (volatilitas & drawdown), Liquidity (volume).")
    st.divider()

    pick = st.selectbox("Detail chart saham:", options=summary["ticker"].tolist())
    st.session_state["last_pick"] = pick

    if pick in data:
        df = data[pick].dropna()
        st.subheader(f"📈 Detail: {pick}")

        col1, col2, col3, col4 = st.columns(4)
        srow = summary[summary["ticker"] == pick].iloc[0].to_dict()
        col1.metric("Health", f"{srow['health_tech']:.1f}/100")
        col2.metric("Close", f"{srow['close']:,.0f}")
        col3.metric("RSI(14)", f"{srow['rsi14']:.1f}")
        col4.metric("Max Drawdown", f"{srow['max_drawdown']:.1%}")

        fig = go.Figure()
        fig.add_candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=pick
        )
        fig.add_scatter(x=df["Date"], y=df["ma50"], name="MA50")
        fig.add_scatter(x=df["Date"], y=df["ma200"], name="MA200")
        fig.update_layout(height=550, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        rsi_fig = go.Figure()
        rsi_fig.add_scatter(x=df["Date"], y=df["rsi14"], name="RSI14")
        rsi_fig.update_layout(height=250)
        st.plotly_chart(rsi_fig, use_container_width=True)

        vol_fig = go.Figure()
        vol_fig.add_bar(x=df["Date"], y=df["Volume"], name="Volume")
        vol_fig.add_scatter(x=df["Date"], y=df["vol_avg20"], name="Avg20 Volume")
        vol_fig.update_layout(height=250)
        st.plotly_chart(vol_fig, use_container_width=True)

    with st.expander("📌 Penjelasan rumus Health Score (teknikal)"):
        st.markdown("""
**Trend (0–100)**  
- Jika harga > MA50 → +50  
- Jika harga > MA200 → +50  

**Risk (0–100)**  
- Volatilitas 20 hari (annualized) makin besar → score turun  
- Max drawdown makin besar → score turun  

**Liquidity (0–100)**  
- volume hari ini dibanding rata-rata 20 hari (vol_ratio)  
- kalau vol_ratio > 1 artinya lebih ramai dari biasanya → score naik  

**Total**
- Health = 40% Trend + 25% Risk + 15% Liquidity  
""")

# =========================
# TAB 2: FUNDAMENTAL
# =========================
with tab_fund:
    st.subheader("🧾 Fundamental (CSV scored)")

    try:
        fund = load_fundamental_csv(fund_path)
    except Exception as e:
        st.error("Gagal membaca CSV fundamental.")
        st.code(str(e))
        st.stop()

    # mapping kolom
    tick_col_guess = infer_ticker_column(fund)
    score_col_guess = infer_score_column(fund)

    with st.expander("🔧 Mapping kolom Fundamental", expanded=True):
        cols = list(fund.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker_col = st.selectbox(
                "Kolom ticker/kode emiten",
                cols,
                index=cols.index(tick_col_guess) if tick_col_guess in cols else 0
            )
        with c2:
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(fund[c])]
            if not num_cols:
                st.error("Tidak ada kolom numerik. Fundamental score harus numerik.")
                st.stop()
            score_col = st.selectbox(
                "Kolom fundamental score",
                num_cols,
                index=num_cols.index(score_col_guess) if score_col_guess in num_cols else 0
            )
        with c3:
            fund_label = st.text_input("Nama score untuk display", value=str(score_col))

    fund_view = fund.copy()
    fund_view["ticker_base"] = fund_view[ticker_col].astype(str).map(to_base_ticker)

    # filter ke saham yang sedang dipilih (opsional)
    only_picked = st.toggle("Tampilkan hanya ticker yang dipilih di sidebar", value=True)
    picked_base = [to_base_ticker(x) for x in st.session_state.get("picked", [])]

    if only_picked and picked_base:
        fund_view = fund_view[fund_view["ticker_base"].isin(picked_base)]

    fund_view = fund_view.sort_values(score_col, ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", f"{len(fund_view):,}")
    k2.metric("Unique emiten", f"{fund_view['ticker_base'].nunique():,}")
    k3.metric("Avg score", f"{fund_view[score_col].mean():.3f}" if len(fund_view) else "-")

    st.dataframe(fund_view, use_container_width=True, height=420)

    if len(fund_view):
        topn = min(20, len(fund_view))
        fig = px.bar(
            fund_view.head(topn).sort_values(score_col),
            x=score_col, y="ticker_base", orientation="h",
            title=f"Top {topn} Fundamental Score"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: COMBINED
# =========================
with tab_combo:
    st.subheader("🧩 Gabungan (Teknikal + Fundamental)")

    # Recompute / re-use technical summary
    data = load_prices(picked, start=str(start))
    rows = []
    for t, df in data.items():
        s = health_from_df(df)
        rows.append({
            "ticker": t,
            "ticker_base": to_base_ticker(t),
            "health_tech": s["health"],
            "trend": s["trend"],
            "risk": s["risk"],
            "liquidity": s["liq"],
        })
    tech_df = pd.DataFrame(rows).dropna(subset=["health_tech"])

    # Load fundamental
    try:
        fund = load_fundamental_csv(fund_path)
    except Exception as e:
        st.error("Gagal membaca CSV fundamental (untuk tab gabungan).")
        st.code(str(e))
        st.stop()

    tick_col_guess = infer_ticker_column(fund)
    score_col_guess = infer_score_column(fund)

    # quick mapper di tab combo (pakai default tebakan biar praktis)
    if tick_col_guess is None or score_col_guess is None:
        st.warning("Aku belum bisa nebak kolom ticker/score fundamental. Buka tab Fundamental untuk mapping manual.")
        st.stop()

    fund2 = fund.copy()
    fund2["ticker_base"] = fund2[tick_col_guess].astype(str).map(to_base_ticker)
    fund2 = fund2.rename(columns={score_col_guess: "score_fund"})

    # join
    combined = tech_df.merge(
        fund2[["ticker_base", "score_fund"]],
        on="ticker_base",
        how="left"
    )

    # Normalisasi fundamental ke 0..100 agar comparable (kalau fundamental sudah 0..100, ini tetap aman)
    if combined["score_fund"].notna().any():
        fmin = float(combined["score_fund"].min())
        fmax = float(combined["score_fund"].max())
        if fmax > fmin:
            combined["fund_norm_0_100"] = (combined["score_fund"] - fmin) / (fmax - fmin) * 100.0
        else:
            combined["fund_norm_0_100"] = np.nan
    else:
        combined["fund_norm_0_100"] = np.nan

    # Combined score
    combined["w_tech"] = w_tech
    combined["w_fund"] = w_fund
    denom = (w_tech + w_fund) if (w_tech + w_fund) > 0 else 1.0

    combined["score_combined"] = (
        (combined["health_tech"] * w_tech) +
        (combined["fund_norm_0_100"] * w_fund)
    ) / denom

    # tampilkan ranking
    show_only_has_fund = st.toggle("Hanya tampilkan yang punya fundamental score", value=False)
    view = combined.copy()
    if show_only_has_fund:
        view = view.dropna(subset=["fund_norm_0_100"])

    view = view.sort_values("score_combined", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah ticker", f"{len(view):,}")
    c2.metric("Ada fundamental", f"{view['fund_norm_0_100'].notna().sum():,}")
    c3.metric("Bobot T/F", f"{w_tech:.2f} / {w_fund:.2f}")
    c4.metric("Avg combined", f"{view['score_combined'].mean():.2f}" if len(view) else "-")

    st.dataframe(
        view[[
            "ticker", "ticker_base",
            "score_combined", "health_tech", "fund_norm_0_100",
            "trend", "risk", "liquidity",
            "score_fund"
        ]].style.format({
            "score_combined": "{:.1f}",
            "health_tech": "{:.1f}",
            "fund_norm_0_100": "{:.1f}",
            "trend": "{:.0f}",
            "risk": "{:.1f}",
            "liquidity": "{:.1f}",
            "score_fund": "{:.3f}",
        }),
        use_container_width=True,
        height=420
    )

    # scatter plot: teknikal vs fundamental
    if view["fund_norm_0_100"].notna().any():
        fig = px.scatter(
            view,
            x="health_tech",
            y="fund_norm_0_100",
            hover_data=["ticker"],
            title="Teknikal vs Fundamental (normalized 0–100)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada fundamental yang match ticker (cek format ticker di CSV: UNVR vs UNVR.JK).")
