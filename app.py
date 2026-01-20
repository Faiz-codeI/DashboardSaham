import streamlit as st

st.set_page_config(
    page_title="Dashboard Kesehatan Saham IDX",
    layout="wide"
)

st.title("📊 Dashboard Kesehatan Saham IDX")

st.success("Streamlit + GitHub sudah terhubung 🚀")

st.write(
    """
    Tahap berikutnya:
    - Load data saham (CSV / Parquet)
    - Pilih emiten
    - Hitung status: Sehat / Waspada / Bahaya
    """
)

