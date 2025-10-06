import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Kalkulator Pensiun Emas + Zakat",
    page_icon="🕌",
    layout="wide"
)

# =========================
# Util & Core Calculations
# =========================

@dataclass
class YearRecord:
    year_idx: int
    start_balance_g: float
    zakat_g: float
    consumption_g: float
    end_balance_g: float

def simulate_retirement_path(
    initial_grams: float,
    years_of_retirement: int,
    annual_consumption_g: float,
    zakat_rate: float = 0.025,
    nisab_g: float = 85.0,
    zakat_first: bool = True
) -> Tuple[List[YearRecord], float, bool]:
    """
    Simulate year-by-year balance reduction with zakat & consumption.
    zakat_first=True means zakat dihitung di awal tahun (setelah haul), lalu konsumsi.
    Returns: (records, final_balance, never_negative_flag)
    """
    records: List[YearRecord] = []
    bal = float(initial_grams)
    never_negative = True

    for y in range(1, years_of_retirement + 1):
        start = bal

        # zakat jika saldo >= nisab
        if zakat_first:
            zakat = (bal * zakat_rate) if bal >= nisab_g else 0.0
            bal -= zakat
            # konsumsi satu tahun
            bal -= annual_consumption_g
        else:
            # konsumsi dulu
            bal -= annual_consumption_g
            # lalu zakat
            zakat = (bal * zakat_rate) if bal >= nisab_g else 0.0
            bal -= zakat

        end = bal
        if end < -1e-9:
            never_negative = False

        records.append(
            YearRecord(
                year_idx=y,
                start_balance_g=start,
                zakat_g=zakat,
                consumption_g=annual_consumption_g,
                end_balance_g=end
            )
        )

    return records, bal, never_negative


def find_required_initial_grams(
    years_of_retirement: int,
    annual_consumption_g: float,
    zakat_rate: float = 0.025,
    nisab_g: float = 85.0,
    zakat_first: bool = True,
    tolerance: float = 1e-4,
    max_iter: int = 200
) -> Tuple[float, List[YearRecord]]:
    """
    Binary search jumlah gram awal agar saldo tidak negatif sepanjang masa pensiun.
    """
    # batas bawah & atas
    lo = 0.0
    # start upper bound: konsumsi total + margin untuk zakat
    hi = max(annual_consumption_g * years_of_retirement * (1 + zakat_rate * 1.2), 1.0)

    # pastikan hi cukup besar
    for _ in range(60):
        recs, final_bal, ok = simulate_retirement_path(
            hi, years_of_retirement, annual_consumption_g, zakat_rate, nisab_g, zakat_first
        )
        if ok and final_bal >= -1e-9:
            break
        hi *= 1.5

    best_records = []
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        recs, final_bal, ok = simulate_retirement_path(
            mid, years_of_retirement, annual_consumption_g, zakat_rate, nisab_g, zakat_first
        )
        if ok and final_bal >= -1e-6:
            best_records = recs
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break

    return hi, best_records


def simulate_accumulation_balance(
    monthly_contribution_g: float,
    months: int,
    zakat_rate: float = 0.025,
    nisab_g: float = 85.0,
    zakat_during_accumulation: bool = False
) -> float:
    """
    Simulasi nabung bulanan (dalam gram) selama 'months'.
    Jika zakat_during_accumulation=True, zakat ditarik di akhir tiap tahun jika >= nisab.
    """
    bal = 0.0
    for m in range(1, months + 1):
        bal += monthly_contribution_g
        if zakat_during_accumulation and (m % 12 == 0):
            if bal >= nisab_g:
                bal -= bal * zakat_rate
    return bal


def find_required_monthly_contribution(
    target_grams: float,
    months: int,
    zakat_rate: float = 0.025,
    nisab_g: float = 85.0,
    zakat_during_accumulation: bool = False,
    tolerance: float = 1e-5,
    max_iter: int = 200
) -> float:
    """
    Cari setoran bulanan (gram) agar balance >= target_grams saat pensiun.
    """
    if months <= 0:
        return float("nan")

    lo = 0.0
    hi = max(target_grams / max(months, 1) * 2.5, 0.1)

    # pastikan hi cukup besar
    for _ in range(60):
        bal = simulate_accumulation_balance(hi, months, zakat_rate, nisab_g, zakat_during_accumulation)
        if bal >= target_grams:
            break
        hi *= 1.6

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        bal = simulate_accumulation_balance(mid, months, zakat_rate, nisab_g, zakat_during_accumulation)
        if bal >= target_grams:
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break
    return hi


def format_g(x: float) -> str:
    return f"{x:,.2f} g".replace(",", "_").replace(".", ",").replace("_", ".")


# =========================
# UI
# =========================

with st.sidebar:
    st.header("⚙️ Input Asumsi")
    colA, colB = st.columns(2)
    with colA:
        curr_age = st.number_input("Usia saat ini", min_value=18, max_value=80, value=46, step=1)
        retire_age = st.number_input("Usia pensiun (target)", min_value=40, max_value=75, value=60, step=1)
        life_expectancy = st.number_input("Usia harapan hidup (tahun)", min_value=60.0, max_value=90.0, value=74.2, step=0.1)
    with colB:
        monthly_need_g = st.number_input("Biaya hidup / bulan (gram emas)", min_value=0.5, max_value=50.0, value=3.0, step=0.1)
        zakat_rate = st.number_input("Tarif zakat emas (%)", min_value=0.0, max_value=5.0, value=2.5, step=0.1) / 100.0
        nisab_g = st.number_input("Nisab emas (gram)", min_value=50.0, max_value=120.0, value=85.0, step=0.5)

    st.divider()
    zakat_during_acc = st.checkbox("Bayar zakat saat masa nabung (sebelum pensiun)?", value=True,
                                   help="Kalau dicentang, tiap akhir tahun saat saldo ≥ nisab akan ditarik 2,5%.")

    safety_buffer_pct = st.slider("Safety buffer saat mulai pensiun (%)", min_value=0, max_value=30, value=10, step=1,
                                  help="Tambahan penyangga untuk antisipasi risiko (kesehatan, kebutuhan naik, dll).")

    st.caption("Tips: Semua satuan berbasis **gram**, biar kebal inflasi harga rupiah.")

st.title("🕌 Kalkulator Pensiun Emas + Zakat (Indonesia)")

if retire_age <= curr_age:
    st.error("Usia pensiun harus lebih besar dari usia saat ini.")
    st.stop()

years_to_retire = retire_age - curr_age
years_retirement = max(0, int(round(life_expectancy - retire_age)))
months_to_retire = years_to_retire * 12
annual_consumption_g = monthly_need_g * 12

st.subheader("📌 Ringkasan Asumsi")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Waktu Nabung", f"{years_to_retire} tahun")
c2.metric("Durasi Pensiun (≈)", f"{years_retirement} tahun")
c3.metric("Konsumsi Tahunan", format_g(annual_consumption_g))
c4.metric("Nisab Emas", format_g(nisab_g))

# Hitung kebutuhan gram awal pensiun (tanpa buffer), include zakat di masa pensiun
base_required_g, base_records = find_required_initial_grams(
    years_of_retirement=years_retirement,
    annual_consumption_g=annual_consumption_g,
    zakat_rate=zakat_rate,
    nisab_g=nisab_g,
    zakat_first=True,
)

buffer_required_g = base_required_g * (1 + safety_buffer_pct / 100.0)

# Hitung kebutuhan nabung bulanan (untuk mencapai target awal pensiun)
monthly_wo_zakat = base_required_g / months_to_retire
monthly_with_zakat = find_required_monthly_contribution(
    target_grams=base_required_g,
    months=months_to_retire,
    zakat_rate=zakat_rate,
    nisab_g=nisab_g,
    zakat_during_accumulation=zakat_during_acc
)

monthly_with_zakat_buffer = find_required_monthly_contribution(
    target_grams=buffer_required_g,
    months=months_to_retire,
    zakat_rate=zakat_rate,
    nisab_g=nisab_g,
    zakat_during_accumulation=zakat_during_acc
)

st.subheader("🎯 Target Gram Emas")
cA, cB, cC = st.columns(3)
cA.metric("Kebutuhan awal pensiun (tanpa buffer)", format_g(base_required_g))
cB.metric(f"Kebutuhan awal + buffer {safety_buffer_pct}%", format_g(buffer_required_g))
cC.metric("Konsumsi total (tanpa zakat)", format_g(annual_consumption_g * years_retirement))

st.subheader("💰 Rekomendasi Nabung Bulanan (gram)")
cX, cY, cZ = st.columns(3)
cX.metric("Tanpa zakat saat nabung", f"{monthly_wo_zakat:,.3f} g/bln".replace(",", "_").replace(".", ",").replace("_", "."))
label_y = "Dengan zakat saat nabung (target tanpa buffer)" if zakat_during_acc else "Setara tanpa zakat saat nabung"
cY.metric(label_y, f"{monthly_with_zakat:,.3f} g/bln".replace(",", "_").replace(".", ",").replace("_", "."))
cZ.metric("Dengan zakat saat nabung (target + buffer)",
          f"{monthly_with_zakat_buffer:,.3f} g/bln".replace(",", "_").replace(".", ",").replace("_", "."))

st.caption("Catatan: Perhitungan mengasumsikan akumulasi murni gram emas (tanpa return investasi), zakat 2.5% per tahun ketika saldo ≥ nisab.")

# =========================
# Tabel & Grafik Jalur Pensiun
# =========================

def records_to_df(records: List[YearRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Tahun ke-": r.year_idx,
        "Saldo Awal (g)": r.start_balance_g,
        "Zakat (g)": r.zakat_g,
        "Konsumsi (g)": r.consumption_g,
        "Saldo Akhir (g)": r.end_balance_g
    } for r in records])

# Simulasikan ulang untuk jalur dengan buffer (biar sesuai target deploy)
records_buffer, _, _ = simulate_retirement_path(
    buffer_required_g, years_retirement, annual_consumption_g, zakat_rate, nisab_g, zakat_first=True
)

tab1, tab2 = st.tabs(["📄 Tabel Simulasi Pensiun", "📈 Grafik Saldo (dengan Buffer)"])

with tab1:
    st.write("### Simulasi tanpa buffer (baseline)")
    df_base = records_to_df(base_records)
    st.dataframe(df_base.style.format({
        "Saldo Awal (g)": "{:,.2f}",
        "Zakat (g)": "{:,.2f}",
        "Konsumsi (g)": "{:,.2f}",
        "Saldo Akhir (g)": "{:,.2f}",
    }), use_container_width=True, height=360)

    st.write("### Simulasi dengan buffer")
    df_buf = records_to_df(records_buffer)
    st.dataframe(df_buf.style.format({
        "Saldo Awal (g)": "{:,.2f}",
        "Zakat (g)": "{:,.2f}",
        "Konsumsi (g)": "{:,.2f}",
        "Saldo Akhir (g)": "{:,.2f}",
    }), use_container_width=True, height=360)

    csv_base = df_base.to_csv(index=False).encode("utf-8")
    csv_buf = df_buf.to_csv(index=False).encode("utf-8")
    colD, colE = st.columns(2)
    colD.download_button("⬇️ Unduh CSV (Baseline)", data=csv_base, file_name="simulasi_pensiun_baseline.csv", mime="text/csv")
    colE.download_button("⬇️ Unduh CSV (Buffer)", data=csv_buf, file_name="simulasi_pensiun_buffer.csv", mime="text/csv")

with tab2:
    import matplotlib.pyplot as plt

    # Plot saldo akhir per tahun (buffer)
    saldo = [rec.start_balance_g for rec in records_buffer]
    saldo.append(records_buffer[-1].end_balance_g if records_buffer else buffer_required_g)

    fig, ax = plt.subplots()
    ax.plot(range(0, len(saldo)), saldo, marker="o")
    ax.set_xlabel("Tahun ke-")
    ax.set_ylabel("Saldo (gram)")
    ax.set_title("Trajektori Saldo Emas Saat Pensiun (Dengan Buffer)")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig, use_container_width=True)

st.divider()
st.markdown("""
### 🧮 Cara Baca Hasil
- **Kebutuhan awal pensiun**: jumlah gram yang perlu siap di usia pensiun agar konsumsi bulanan terpenuhi **+** zakat tahunan (jika ≥ nisab) sepanjang durasi pensiun.
- **Nabung bulanan**: gram per bulan yang perlu dikumpulkan mulai sekarang hingga usia pensiun. Opsi *zakat saat nabung* mensimulasikan 2,5% per tahun (tiap 12 bulan) ketika saldo ≥ nisab.
- **Buffer**: penyangga ekstra supaya aman kalau durasi pensiun lebih panjang/biaya naik.

> Best practice: tetap gunakan **DCA** (beli rutin bulanan dalam gram), bayar zakat **pakai gram** atau rupiah setara harga saat bayar, dan review asumsi tiap tahun.
""")

st.caption("© 2025 — Dibuat untuk Bro Herman. Semoga bermanfaat. Insya Allah bisa 🙏")
