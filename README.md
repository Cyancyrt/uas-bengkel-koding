# CareerSignal — Prediksi Keinginan Pindah Kerja

Aplikasi Streamlit untuk proyek UAS Bengkel Koding Data Science 2026/2027. Menggunakan Random Forest yang dipilih berdasarkan F1 OOF notebook Optimasi Terkontrol dengan threshold **0.5233544005173285**. Model sudah dilatih dan disimpan bersama pipeline; tidak perlu training saat aplikasi dibuka.

## Menjalankan lokal

Gunakan Python 3.12 agar sesuai dengan lingkungan pengujian. Buka terminal pada folder ini:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deployment Streamlit Community Cloud

1. Buat repository GitHub dan upload isi folder ini (termasuk `artifacts/`, `.streamlit/`, `notebook/`, `requirements.txt`). Jangan hanya upload `app.py`.
2. Buka https://share.streamlit.io/ lalu pilih **Create app**.
3. Pilih repository, branch, dan main file `app.py`. Gunakan Python 3.12 pada pengaturan jika tersedia.
4. Klik Deploy, tunggu dependensi terpasang, lalu uji profil simulasi dan tab evaluasi. Masukkan link publik Streamlit pada pengumpulan UAS.

## Struktur

- `app.py`: antarmuka prediksi, evaluasi dan metodologi.
- `model_pipeline.py`: fungsi/kelas preprocessing dan Random Forest portabel untuk joblib.
- `artifacts/model.joblib`: model dan preprocessing yang sudah dilatih.
- `artifacts/metadata.json`: schema input, profil simulasi, threshold dan hasil evaluasi model yang diserialisasi.
- `artifacts/comparison.csv`, `artifacts/importance.csv`: hasil notebook.
- `train_model.py`: reproduksi konfigurasi akhir dari CSV asli; tidak melakukan tuning ulang.
- `test_app.py`: pengujian fungsional form, kedua kelas prediksi, input kosong, dan konsistensi model (`python test_app.py`).
- `notebook/`: notebook UAS berisi eksperimen yang telah dieksekusi.

## Reproduksi model (opsional)

Dataset asli tidak disertakan dalam paket aplikasi; letakkan `UAS_DS_BENGKOD.csv` pada folder ini atau masukkan path lengkap. Jalankan:

```bash
python train_model.py UAS_DS_BENGKOD.csv
```

Model dipasang pada data latih 80%, oversampling hanya pada train. Hasil evaluasi disimpan di metadata. Konfigurasi sama dengan notebook; versi scikit-learn runtime deployment adalah 1.8.0 (notebook Colab memakai 1.6.1). Jangan mencampur versi scikit-learn saat load joblib. Profil contoh bersifat sintetis. Tidak ada data peserta atau input aplikasi disimpan di aplikasi.

## Demonstrasi

1. Buka tab Prediksi peserta, pilih profil simulasi Karier stabil atau Eksplorasi karier.
2. Tekan Jalankan prediksi. Jelaskan label 0/1, skor model, dan threshold OOF.
3. Buka Evaluasi model untuk confusion matrix, perbandingan 9 eksperimen, dan feature importance.
4. Jelaskan bahwa skor model bukan peluang terkalibrasi dan fitur berisiko leakage sengaja dikeluarkan.

**Catatan:** model untuk tujuan pembelajaran, bukan penilaian otomatis terhadap karyawan atau calon pelamar. Aplikasi tidak meminta ID peserta, status resign, kode departemen, atau aktivitas melamar LinkedIn.
