import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ======================
# Page & Global Styling
# ======================
st.set_page_config(
    page_title="Kalkulator Pensiun Emas + Zakat",
    page_icon="🕌",
    layout="wide"
)

# --- Minimalist CSS ---
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
.section-title {font-weight: 700; font-size: 1.05rem; margin: .1rem 0 .8rem;}
.card {
  padding: 1rem 1.2rem; border: 1px solid #e5e7eb; border-radius: 14px;
  background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.04); margin-bottom: 1rem;
}
.kpi {padding:.8rem .9rem;border-radius:12px;background:#f8fafc;border:1px solid #eef2f7;text-align:center}
.kpi .label {font-size:.82rem;color:#64748b;margin-bottom:.25rem}
.kpi .value {font-size:1.25rem;font-weight:700;color:#0f172a}
.small {font-size:.86rem;color:#64748b}
[data-baseweb="tab-content"] {padding-top: .5rem;}
/* Sidebar grouping */
.stSidebar .stNumberInput, .stSidebar .stSlider, .stSidebar .stSelectbox {margin-bottom: .6rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Baris sidebar yang rapat */
.stSidebar .srow { margin: .10rem 0; }                 /* jarak antar baris */
.stSidebar .slabel { font-weight: 500; color:#111827; }

/* Kecilkan tinggi number input & hilangkan margin default */
.stSidebar [data-testid="stNumberInput"]{ margin:0 !important; }
.stSidebar [data-testid="stNumberInput"] input{ height:30px; padding:.15rem .5rem; text-align:right; }
.stSidebar [data-testid="stNumberInput"] button{ height:30px; }
</style>
""", unsafe_allow_html=True)



# =========================
# Utils & Core Calcs
# =========================
@dataclass
class YearRecord:
    year_idx: int
    start_balance_g: float
    zakat_g: float
    consumption_g: float
    end_balance_g: float

@dataclass
class AccMonthRecord:
    month_idx: int
    start_balance_g: float
    contribution_g: float
    zakat_g: float
    end_balance_g: float
    zakat_month: bool
    
def num_row(label: str, *, key: str, value, min_value, max_value, step, fmt=None):
    # container satu baris → 2 kolom agar selalu sejajar
    with st.sidebar.container():
        c1, c2 = st.columns([1, 0.55], vertical_alignment="center")
        with c1:
            st.markdown(f'<div class="srow slabel">{label}</div>', unsafe_allow_html=True)
        with c2:
            return st.number_input(
                label, key=key, value=value,
                min_value=min_value, max_value=max_value, step=step,
                format=fmt if fmt else None,
                label_visibility="collapsed",
            )


def format_g(x: float) -> str:
    return f"{x:,.2f} g".replace(",", "_").replace(".", ",").replace("_", ".")

def format_rp(x: float) -> str:
    try:
        return "Rp " + f"{x:,.0f}".replace(",", ".")
    except Exception:
        return "Rp -"

def kpi(label: str, value: str):
    st.markdown(f"""
    <div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>
    """, unsafe_allow_html=True)

# =========================
# Harga Emas (multi-source, cached)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_gold_prices() -> dict:
    """
    Return dict: {'sell': float|None, 'buyback': float|None, 'provider': str}
    Sumber berurutan: logammulia.com (harga 1 gr & buyback) → anekalogam.co.id → API komunitas.
    """
    import re, requests
    from bs4 import BeautifulSoup

    def rupiah_to_float(s: str) -> float:
        return float(re.sub(r"[^0-9]", "", s))

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    sell = buyback = None
    provider = ""

    # --- Sumber 1: Antam (harga 1 gr)
    try:
        r = requests.get("https://www.logammulia.com/harga-emas-hari-ini", headers=headers, timeout=12)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        m = re.search(r"\b1\s*gr\b[^0-9]*([0-9\.\,]+)", text, flags=re.I)
        if m:
            sell = rupiah_to_float(m.group(1))
            provider = "logammulia.com (1 gr)"
    except Exception:
        pass

    # --- Sumber 1b: Antam (buyback)
    try:
        r = requests.get("https://www.logammulia.com/id/sell/gold", headers=headers, timeout=12)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        m = re.search(r"Harga\s*Buyback[^0-9]*Rp\s*([0-9\.\,]+)", text, flags=re.I)
        if m:
            buyback = rupiah_to_float(m.group(1))
            if not provider:
                provider = "logammulia.com (buyback)"
    except Exception:
        pass

    # --- Sumber 2: AnekaLogam (fallback)
    if sell is None or buyback is None:
        try:
            r = requests.get("https://anekalogam.co.id/id", headers=headers, timeout=12)
            r.raise_for_status()
            text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
            if sell is None:
                m = re.search(r"Harga\s*Jual[^0-9]*Rp\s*([0-9\.\,]+)", text, flags=re.I)
                if m: sell = rupiah_to_float(m.group(1))
            if buyback is None:
                m = re.search(r"Harga\s*Beli[^0-9]*Rp\s*([0-9\.\,]+)", text, flags=re.I)
                if m: buyback = rupiah_to_float(m.group(1))
            if (sell is not None) or (buyback is not None):
                provider = "anekalogam.co.id"
        except Exception:
            pass

    # --- Sumber 3: API komunitas (fallback terakhir)
    if sell is None or buyback is None:
        try:
            r = requests.get("https://logam-mulia-api.vercel.app/prices/anekalogam", timeout=12)
            r.raise_for_status()
            js = r.json()
            if js and "data" in js and js["data"]:
                item = js["data"][0]
                # key umum: 'sel' (jual) dan 'buy' (buyback)
                if sell is None and item.get("sel"):
                    sell = float(item["sel"])
                if buyback is None and item.get("buy"):
                    buyback = float(item["buy"])
                provider = "logam-mulia-api (anekalogam)"
        except Exception:
            pass

    return {"sell": sell, "buyback": buyback, "provider": provider}

# =========================
# Pensiun phase
# =========================
def simulate_retirement_path(
    initial_grams: float, years_of_retirement: int, annual_consumption_g: float,
    zakat_rate: float = 0.025, nisab_g: float = 85.0, zakat_first: bool = True
) -> Tuple[List[YearRecord], float, bool]:
    records: List[YearRecord] = []
    bal = float(initial_grams); never_negative = True
    for y in range(1, years_of_retirement + 1):
        start = bal
        if zakat_first:
            zakat = bal * zakat_rate if bal >= nisab_g else 0.0
            bal -= zakat; bal -= annual_consumption_g
        else:
            bal -= annual_consumption_g
            zakat = bal * zakat_rate if bal >= nisab_g else 0.0
            bal -= zakat
        end = bal
        if end < -1e-9: never_negative = False
        records.append(YearRecord(y, start, zakat, annual_consumption_g, end))
    return records, bal, never_negative

def find_required_initial_grams(
    years_of_retirement: int, annual_consumption_g: float,
    zakat_rate: float = 0.025, nisab_g: float = 85.0, zakat_first: bool = True,
    tolerance: float = 1e-4, max_iter: int = 200
) -> Tuple[float, List[YearRecord]]:
    lo = 0.0
    hi = max(annual_consumption_g * years_of_retirement * (1 + zakat_rate * 1.2), 1.0)
    for _ in range(60):
        recs, final_bal, ok = simulate_retirement_path(hi, years_of_retirement, annual_consumption_g,
                                                       zakat_rate, nisab_g, zakat_first)
        if ok and final_bal >= -1e-9: break
        hi *= 1.5

    best_records = []
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        recs, final_bal, ok = simulate_retirement_path(mid, years_of_retirement, annual_consumption_g,
                                                       zakat_rate, nisab_g, zakat_first)
        if ok and final_bal >= -1e-6:
            best_records = recs; hi = mid
        else:
            lo = mid
        if hi - lo < tolerance: break
    return hi, best_records

# =========================
# Accumulation phase (start balance & escalation)
# =========================
def simulate_accumulation_path(
    base_monthly_contribution_g: float, months: int, start_balance_g: float = 0.0,
    escalation_rate_per_year: float = 0.0, zakat_rate: float = 0.025,
    nisab_g: float = 85.0, zakat_during_accumulation: bool = True
) -> Tuple[List[AccMonthRecord], float, float, float]:
    records: List[AccMonthRecord] = []; bal = float(start_balance_g)
    total_zakat = 0.0; total_contrib = 0.0
    for m in range(1, months + 1):
        start = bal
        year_idx = (m - 1) // 12
        contrib = base_monthly_contribution_g * ((1.0 + escalation_rate_per_year) ** year_idx)
        bal += contrib; total_contrib += contrib
        zakat = 0.0; zakat_flag = False
        if zakat_during_accumulation and (m % 12 == 0) and (bal >= nisab_g):
            zakat = bal * zakat_rate; bal -= zakat; total_zakat += zakat; zakat_flag = True
        records.append(AccMonthRecord(m, start, contrib, zakat, bal, zakat_flag))
    return records, bal, total_zakat, total_contrib

def find_required_base_monthly_for_target(
    target_grams: float, months: int, start_balance_g: float = 0.0,
    escalation_rate_per_year: float = 0.0, zakat_rate: float = 0.025,
    nisab_g: float = 85.0, zakat_during_accumulation: bool = True,
    tolerance: float = 1e-5, max_iter: int = 200
) -> float:
    if months <= 0: return 0.0
    est_plain = max(target_grams - start_balance_g, 0.0) / max(months, 1)
    lo, hi = 0.0, max(est_plain * 2.5, 0.1)
    for _ in range(60):
        _, bal, _, _ = simulate_accumulation_path(
            hi, months, start_balance_g, escalation_rate_per_year, zakat_rate, nisab_g, zakat_during_accumulation)
        if bal >= target_grams: break
        hi *= 1.6
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        _, bal, _, _ = simulate_accumulation_path(
            mid, months, start_balance_g, escalation_rate_per_year, zakat_rate, nisab_g, zakat_during_accumulation)
        if bal >= target_grams: hi = mid
        else: lo = mid
        if hi - lo < tolerance: break
    return hi

# =========================
# Sidebar (urut sesuai gambar)
# =========================
with st.sidebar:
    st.markdown("<div class='section-title'>⚙️ Input Asumsi</div>", unsafe_allow_html=True)

    curr_age   = num_row("Usia saat ini",             key="age_now",  value=46, min_value=18,  max_value=80,  step=1)
    retire_age = num_row("Usia pensiun (target)",     key="age_ret",  value=55, min_value=40,  max_value=75,  step=1)
    monthly_need_g = num_row("Biaya hidup / bulan (gram)", key="need_g",
                             value=3.00, min_value=0.50, max_value=50.0, step=0.10, fmt="%.2f")

    # jeda kecil
    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)

    life_expectancy = num_row("Usia harapan hidup (tahun)", key="life_exp",
                              value=72.0, min_value=60.0, max_value=90.0, step=0.1, fmt="%.1f")
    zakat_rate_pct  = num_row("Tarif zakat emas (%)",       key="zakat_pct",
                              value=2.5, min_value=0.0, max_value=5.0, step=0.1, fmt="%.1f")
    nisab_g         = num_row("Nisab emas (gram)",          key="nisab",
                              value=85.0, min_value=50.0, max_value=120.0, step=0.5, fmt="%.0f")
    zakat_rate = zakat_rate_pct / 100.0

    st.markdown("<div class='section-title'>📦 Saldo & Penyetoran</div>", unsafe_allow_html=True)
    current_gold_g = num_row("Saldo emas saat ini (gram)",  key="saldo_now",
                             value=0.0, min_value=0.0, max_value=5000.0, step=0.1, fmt="%.2f")

    annual_escalation_pct = st.slider(
        "Kenaikan setoran per tahun (%)", min_value=0, max_value=50, value=0, step=1,
        help="0% = setoran bulanan tetap; 10% = naik 10% tiap tahun."
    )

    with st.expander("Advanced options", expanded=False):
        zakat_during_acc = st.checkbox("Bayar zakat saat masa nabung (tiap 12 bulan jika ≥ nisab)?", value=True)
        safety_buffer_pct = st.slider("Safety buffer saat mulai pensiun (%)", min_value=0, max_value=30, value=10, step=1)



st.markdown("## 🕌 Kalkulator Pensiun Emas + Zakat (Indonesia)")

if retire_age <= curr_age:
    st.error("Usia pensiun harus lebih besar dari usia saat ini.")
    st.stop()

years_to_retire = retire_age - curr_age
years_retirement = max(0, int(round(life_expectancy - retire_age)))
months_to_retire = years_to_retire * 12
annual_consumption_g = monthly_need_g * 12
esc_rate = annual_escalation_pct / 100.0

# =========================
# Harga Emas (UI + Refresh)
# =========================
st.markdown('<div class="section-title">💰 Harga Emas Antam Hari Ini</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

# Tombol refresh cache
cbtn, _, _ = st.columns([1,1,1])
with cbtn:
    if st.button("🔄 Refresh harga (clear cache)"):
        st.cache_data.clear()

prices = fetch_gold_prices()
harga_jual_antam = prices["sell"]
buyback_antam = prices["buyback"]
provider_used = prices["provider"] or "—"

colP1, colP2, colP3 = st.columns([1,1,1])
with colP1:
    kpi("Antam - Harga Jual / gram", format_rp(harga_jual_antam) if harga_jual_antam else "Gagal fetch")
with colP2:
    kpi("Antam - Buyback / gram", format_rp(buyback_antam) if buyback_antam else "Gagal fetch")
with colP3:
    default_index = 0 if harga_jual_antam else (1 if buyback_antam else 2)
    price_mode = st.selectbox(
        "Gunakan harga untuk konversi rupiah",
        options=["Harga Jual Antam", "Buyback Antam", "Input Manual"],
        index=default_index
    )
manual_price = None
if price_mode == "Input Manual":
    manual_price = st.number_input("Masukkan harga per gram (Rp)", min_value=0.0, max_value=10_000_000.0,
                                   value=1_250_000.0, step=1_000.0)

st.caption(f"Sumber aktif: **{provider_used}** — cache 10 menit. Jika situs diblokir proteksi, gunakan Input Manual.")
st.markdown('</div>', unsafe_allow_html=True)

def get_active_price_per_gram():
    if price_mode == "Harga Jual Antam" and harga_jual_antam:
        return harga_jual_antam
    if price_mode == "Buyback Antam" and buyback_antam:
        return buyback_antam
    return manual_price

active_price = get_active_price_per_gram()

# =========================
# Ringkasan Asumsi (KPI)
# =========================
st.markdown('<div class="section-title">📌 Ringkasan Asumsi</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: kpi("Usia Sekarang", f"{curr_age} th")
with c2: kpi("Usia Pensiun", f"{retire_age} th")
with c3: kpi("Durasi Nabung", f"{years_to_retire} tahun")
with c4: kpi("Durasi Pensiun (≈)", f"{years_retirement} tahun")
with c5: kpi("Konsumsi Tahunan", format_g(annual_consumption_g))
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Kebutuhan gram
# =========================
base_required_g, base_records = find_required_initial_grams(
    years_of_retirement=years_retirement,
    annual_consumption_g=annual_consumption_g,
    zakat_rate=zakat_rate, nisab_g=nisab_g, zakat_first=True,
)
buffer_required_g = base_required_g * (1 + safety_buffer_pct / 100.0)

# Base setoran (dengan saldo awal + zakat + eskalasi)
monthly_with_settings = find_required_base_monthly_for_target(
    target_grams=base_required_g, months=months_to_retire, start_balance_g=current_gold_g,
    escalation_rate_per_year=esc_rate, zakat_rate=zakat_rate, nisab_g=nisab_g,
    zakat_during_accumulation=zakat_during_acc
)
monthly_with_settings_buffer = find_required_base_monthly_for_target(
    target_grams=buffer_required_g, months=months_to_retire, start_balance_g=current_gold_g,
    escalation_rate_per_year=esc_rate, zakat_rate=zakat_rate, nisab_g=nisab_g,
    zakat_during_accumulation=zakat_during_acc
)

# =========================
# Target Gram + Konversi Rp
# =========================
st.markdown('<div class="section-title">🎯 Target Gram Emas & Estimasi Rupiah</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
cA, cB, cC = st.columns(3)
with cA:
    kpi("Kebutuhan awal pensiun (tanpa buffer)", format_g(base_required_g))
    if active_price: st.caption(f"≈ {format_rp(base_required_g * active_price)}")
with cB:
    kpi(f"Kebutuhan awal + buffer {int(safety_buffer_pct)}%", format_g(buffer_required_g))
    if active_price: st.caption(f"≈ {format_rp(buffer_required_g * active_price)}")
with cC:
    kpi("Konsumsi total (tanpa zakat)", format_g(annual_consumption_g * years_retirement))
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Rekomendasi Nabung Bulanan (gram & Rp)
# =========================
st.markdown('<div class="section-title">💰 Rekomendasi Nabung Bulanan</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
cX, cY = st.columns(2)
with cX:
    kpi("Base setoran/bln (target tanpa buffer)",
        f"{monthly_with_settings:,.3f} g".replace(",", "_").replace(".", ",").replace("_", "."))
    if active_price: st.caption(f"≈ {format_rp(monthly_with_settings * active_price)} (bulan 1)")
with cY:
    kpi("Base setoran/bln (target + buffer)",
        f"{monthly_with_settings_buffer:,.3f} g".replace(",", "_").replace(".", ",").replace("_", "."))
    if active_price: st.caption(f"≈ {format_rp(monthly_with_settings_buffer * active_price)} (bulan 1)")
st.caption("Catatan: bila ada eskalasi tahunan, setoran tahun ke-2 = base×(1+r), dst.")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Simulasi Nabung Sebelum Pensiun
# =========================
st.markdown('<div class="section-title">🧱 Simulasi Nabung Sebelum Pensiun</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col_sim1, col_sim2, col_sim3 = st.columns([1,1,1])
with col_sim1:
    target_opt = st.selectbox("Target yang ingin dikejar", ["Tanpa buffer", "Dengan buffer"], index=0)
with col_sim2:
    default_plan = monthly_with_settings if target_opt == "Tanpa buffer" else monthly_with_settings_buffer
    plan_monthly_g = st.number_input("Base setoran/bln (gram)", min_value=0.0, max_value=100.0,
                                     value=float(round(default_plan, 3)), step=0.01,
                                     help="Tahun 1 = base; Tahun 2 = base×(1+r); dst.")
with col_sim3:
    kpi("Periode nabung", f"{months_to_retire} bulan")

target_grams = base_required_g if target_opt == "Tanpa buffer" else buffer_required_g
acc_records, acc_final_bal, total_zakat_acc, total_contribution = simulate_accumulation_path(
    base_monthly_contribution_g=plan_monthly_g, months=months_to_retire, start_balance_g=current_gold_g,
    escalation_rate_per_year=esc_rate, zakat_rate=zakat_rate, nisab_g=nisab_g, zakat_during_accumulation=zakat_during_acc
)
gap = acc_final_bal - target_grams
status = "✅ Tercapai (surplus)" if gap >= 0 else "❌ Belum tercapai (defisit)"

csi1, csi2, csi3, csi4 = st.columns(4)
with csi1: kpi("Target di usia pensiun", format_g(target_grams))
with csi2: kpi("Saldo saat pensiun (hasil simulasi)", format_g(acc_final_bal))
with csi3: kpi("Selisih (surplus/defisit)", format_g(gap))
with csi4: kpi("Total zakat selama nabung", format_g(total_zakat_acc))
st.caption(f"Hasil: **{status}**. Total setoran {format_g(total_contribution)} dalam {months_to_retire} bulan. Saldo awal dihitung: {format_g(current_gold_g)}.")
if active_price:
    st.caption(f"Estimasi nilai saldo akhir saat pensiun ≈ {format_rp(acc_final_bal * active_price)}.")

def acc_records_to_df(recs: List[AccMonthRecord]) -> pd.DataFrame:
    rows = []
    for r in recs:
        usia_akhir_bulan = curr_age + (r.month_idx / 12.0)
        usia_bulat = int(math.floor(usia_akhir_bulan))  # dibulatkan ke bawah
        rows.append({
            "Bulan ke-": r.month_idx,
            "Usia (thn)": usia_bulat,                     # integer
            "Saldo Awal (g)": r.start_balance_g,
            "Setoran (g)": r.contribution_g,
            "Zakat (g)": r.zakat_g,
            "Saldo Akhir (g)": r.end_balance_g,
            "Zakat?": "Ya" if r.zakat_month else "Tidak"
        })
    return pd.DataFrame(rows)

def acc_records_yearly_df(recs: List[AccMonthRecord]) -> pd.DataFrame:
    rows = []
    total_months = len(recs)
    years = (total_months + 11) // 12
    for y in range(years):
        start_idx = y * 12
        end_idx = min((y + 1) * 12, total_months)
        chunk = recs[start_idx:end_idx]
        start_bal = chunk[0].start_balance_g
        end_bal = chunk[-1].end_balance_g
        contrib_sum = sum(c.contribution_g for c in chunk)
        zakat_sum = sum(c.zakat_g for c in chunk)
        months_in_year = end_idx - start_idx

        usia_awal = curr_age + y
        usia_akhir = curr_age + y + months_in_year / 12.0

        # 👉 dibulatkan (floor) ke bilangan bulat
        usia_awal_int = int(math.floor(usia_awal))
        usia_akhir_int = int(math.floor(usia_akhir))

        rows.append({
            "Tahun ke-": y + 1,
            "Usia Awal (thn)": usia_awal_int,
            "Usia Akhir (thn)": usia_akhir_int,
            "Saldo Awal (g)": start_bal,
            "Total Setoran (g)": contrib_sum,
            "Total Zakat (g)": zakat_sum,
            "Saldo Akhir (g)": end_bal
        })
    return pd.DataFrame(rows)


tab_acc1, tab_acc2, tab_acc3 = st.tabs(["📄 Tabel Bulanan", "📄 Tabel Tahunan", "📈 Grafik Saldo Nabung"])

with tab_acc1:
    df_acc = acc_records_to_df(acc_records)

    # ---- Styler: bold seluruh baris saat zakat, plus highlight lembut ----
    def highlight_zakat(row):
        if row["Zakat?"] == "Ya":
            return ["font-weight:700; background-color:#FFF4D6;"] * len(row)
        return [""] * len(row)

    styled = (
        df_acc.style
        .format({
            "Saldo Awal (g)": "{:,.2f}",
            "Setoran (g)": "{:,.2f}",
            "Zakat (g)": "{:,.2f}",
            "Saldo Akhir (g)": "{:,.2f}",
        })
        .apply(highlight_zakat, axis=1)
    )

    # Render sebagai HTML (stabil, tidak memicu React error)
    st.markdown(styled.to_html(), unsafe_allow_html=True)

    # Tombol unduh CSV tetap ada
    st.download_button(
        "⬇️ Unduh CSV (Bulanan)",
        df_acc.to_csv(index=False).encode("utf-8"),
        file_name="simulasi_nabung_bulanan.csv",
        mime="text/csv",
    )

with tab_acc2:
    df_year = acc_records_yearly_df(acc_records)
    st.dataframe(
        df_year,
        use_container_width=True,
        height=360,
        hide_index=True,
        column_config={
            "Usia Awal (thn)": st.column_config.NumberColumn(format="%d"),
            "Usia Akhir (thn)": st.column_config.NumberColumn(format="%d"),
            "Saldo Awal (g)": st.column_config.NumberColumn(format="%.2f"),
            "Total Setoran (g)": st.column_config.NumberColumn(format="%.2f"),
            "Total Zakat (g)": st.column_config.NumberColumn(format="%.2f"),
            "Saldo Akhir (g)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.download_button(
        "⬇️ Unduh CSV (Tahunan)",
        df_year.to_csv(index=False).encode("utf-8"),
        file_name="simulasi_nabung_tahunan.csv",
        mime="text/csv",
    )

with tab_acc3:
    import matplotlib.pyplot as plt
    saldo_m = [r.end_balance_g for r in acc_records]
    fig, ax = plt.subplots()
    ax.plot(range(1, len(saldo_m)+1), saldo_m, marker="o")
    ax.axhline(base_required_g, linestyle="--", alpha=0.6)
    ax.axhline(buffer_required_g, linestyle="--", alpha=0.6)
    ax.set_xlabel("Bulan ke-"); ax.set_ylabel("Saldo (gram)")
    ax.set_title("Akumulasi Saldo Emas Sebelum Pensiun")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Masa Pensiun (tabel & grafik)
# =========================
def pension_records_to_df(records: List[YearRecord], retire_age_base: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "Tahun ke-": r.year_idx, "Usia Awal (thn)": retire_age_base + (r.year_idx - 1),
        "Usia Akhir (thn)": retire_age_base + r.year_idx, "Saldo Awal (g)": r.start_balance_g,
        "Zakat (g)": r.zakat_g, "Konsumsi (g)": r.consumption_g, "Saldo Akhir (g)": r.end_balance_g
    } for r in records])

records_buffer, _, _ = simulate_retirement_path(
    buffer_required_g, years_retirement, annual_consumption_g, zakat_rate, nisab_g, zakat_first=True
)

tab1, tab2 = st.tabs(["📄 Tabel Masa Pensiun", "📈 Grafik Saldo (Masa Pensiun, Buffer)"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ===== Helper render HTML table (aman, tanpa React grid) =====
    def render_html_table(df, number_cols_2dec=None, int_cols=None, height_px=320):
        number_cols_2dec = number_cols_2dec or []
        int_cols = int_cols or []
        if not df.empty:
            for c in int_cols:
                if c in df.columns:
                    df[c] = df[c].astype(int)
        style = df.style.format({c: "{:,.2f}" for c in number_cols_2dec})
        html = style.to_html()
        st.markdown(f"<div style='max-height:{height_px}px; overflow:auto'>{html}</div>", unsafe_allow_html=True)

    # ===== Baseline =====
    st.markdown("#### Simulasi tanpa buffer (baseline)")
    df_base = pension_records_to_df(base_records, retire_age)
    render_html_table(
        df_base,
        number_cols_2dec=["Saldo Awal (g)", "Zakat (g)", "Konsumsi (g)", "Saldo Akhir (g)"],
        int_cols=["Usia Awal (thn)", "Usia Akhir (thn)"],
        height_px=320
    )

    # ===== Buffer =====
    st.markdown("#### Simulasi dengan buffer")
    df_buf = pension_records_to_df(records_buffer, retire_age)
    render_html_table(
        df_buf,
        number_cols_2dec=["Saldo Awal (g)", "Zakat (g)", "Konsumsi (g)", "Saldo Akhir (g)"],
        int_cols=["Usia Awal (thn)", "Usia Akhir (thn)"],
        height_px=320
    )

    # tombol unduh (tetap)
    colD, colE = st.columns(2)
    colD.download_button(
        "⬇️ Unduh CSV (Baseline)",
        df_base.to_csv(index=False).encode("utf-8"),
        file_name="simulasi_pensiun_baseline.csv",
        mime="text/csv",
    )
    colE.download_button(
        "⬇️ Unduh CSV (Buffer)",
        df_buf.to_csv(index=False).encode("utf-8"),
        file_name="simulasi_pensiun_buffer.csv",
        mime="text/csv",
    )

    st.markdown('</div>', unsafe_allow_html=True)



with tab2:
    import matplotlib.pyplot as plt
    st.markdown('<div class="card">', unsafe_allow_html=True)
    saldo = [rec.start_balance_g for rec in records_buffer]
    saldo.append(records_buffer[-1].end_balance_g if records_buffer else buffer_required_g)
    fig, ax = plt.subplots()
    ax.plot(range(0, len(saldo)), saldo, marker="o")
    ax.set_xlabel("Tahun ke-"); ax.set_ylabel("Saldo (gram)")
    ax.set_title("Trajektori Saldo Emas Saat Pensiun (Dengan Buffer)")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="small">
- Harga diambil otomatis & di-cache 10 menit. Bila gagal karena proteksi, klik <em>Refresh</em> atau gunakan <em>Input Manual</em>.
- Konversi rupiah memakai pilihan harga aktif (Jual/Buyback/Manual).
- Setoran bulanan memperhitungkan <em>saldo awal</em> dan <em>kenaikan setoran per tahun</em>.
</div>
""", unsafe_allow_html=True)

st.caption("© 2025 — Dibuat untuk Bro Herman. Insya Allah bisa 🙏")
