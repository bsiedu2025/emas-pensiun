# 🕌 Kalkulator Pensiun Emas + Zakat

Aplikasi Streamlit buat ngitung target **pensiun berbasis emas** (gram) plus **zakat emas** otomatis (nisab & 2.5%/tahun). Fokus di skenario Indonesia.

## Fitur
- Input usia sekarang, usia pensiun, harapan hidup, kebutuhan bulanan (gram)
- Nisab emas & tarif zakat dapat diubah
- Simulasi **tahun per tahun**: zakat → konsumsi
- Hitung **kebutuhan gram awal** di usia pensiun (dengan/ tanpa buffer)
- Hitung **tabungan bulanan** (opsional zakat selama masa nabung)
- Tabel & grafik saldo, **download CSV**

## Jalankan Lokal
```bash
# (opsional) buat virtualenv
pip install -r requirements.txt
streamlit run app.py