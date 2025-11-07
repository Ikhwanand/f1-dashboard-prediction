# F1 Dashboard Prediction

Aplikasi Streamlit untuk memprediksi posisi finish dan perolehan poin pembalap F1 berdasarkan metrik telemetry balapan. Antarmuka ini dirancang sebagai companion dashboard: kamu bisa melakukan eksperimen skenario tunggal, batch inference dari CSV, hingga meninjau kartu model lengkap dengan metrik evaluasi.

## ✨ Fitur Utama

- **Dual-model inference**: SVR untuk prediksi posisi finish dan Extra Trees untuk estimasi poin.
- **Single vs batch prediction**: Masukkan telemetry manual via slider atau unggah CSV dengan ratusan skenario sekaligus.
- **Model cards tersemat**: Pipeline steps, fitur yang dipakai, hyperparameter terbaik, dan metrik performa tampil langsung di UI.
- **Data guardrails**: Kolom yang hilang otomatis diisi nilai default, plus validasi tipe numerik.
- **Siap di-embed**: Cukup jalankan `streamlit run app/main.py` lalu tambahkan komponen hasilnya ke dashboard analitik favoritmu.

## 🗂️ Struktur Proyek

f1-dashboard-prediction/
├── app/
│ └── main.py # Entry-point Streamlit + UI/logic inference
├── models/
│ ├── f1_position_model_svr.pkl
│ ├── f1_points_model_extra trees.pkl
│ └── f1_models_metadata.json
├── notebooks/
│ └── nb-1.ipynb # Eksperimen/training model (tidak dibersihkan untuk prod)
├── requirements.txt # Seluruh dependency (streamlit, scikit-learn, FastF1, dsb.)
└── venv/ # (Opsional) virtual environment lokal



> Catatan: dataset mentah dan artefak ETL tidak dibundel. Pastikan kamu punya salinan lokal ketika ingin melatih ulang.

## 🚀 Persiapan Lingkungan

1. **Gunakan Python 3.10+** agar kompatibel dengan versi `pandas`, `scikit-learn`, dan `streamlit` yang dikunci.
2. (Opsional) buat virtual env:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
Instal dependensi minimal untuk inference:
bash

pip install -r requirements.txt
Jika hanya ingin menjalankan dashboard, kamu bisa menginstal subset (streamlit, pandas, scikit-learn, joblib, fastf1) untuk mempercepat setup.

▶️ Cara Menjalankan Dashboard
bash

streamlit run app/main.py
Streamlit akan terbuka di http://localhost:8501.
Periksa panel samping untuk ringkasan cara pakai dan command cepat.
Tiga tab utama:
Single Prediction – slider per fitur telemetry.
Batch Prediction – unggah CSV dan unduh hasil prediksi.
Model Cards – detail artefak serta metrik training.
Mode Single Prediction
Atur slider sesuai kondisi lap/pembalap.
Klik Predict scenario untuk menjalankan kedua model sekaligus.
Hasil langsung tampil dalam bentuk metrik dan tabel payload fitur (berguna untuk audit).
Mode Batch Prediction
Siapkan CSV dengan kolom bernama sama persis seperti saat training (lihat tabel fitur di bawah).
Unggah file di tab Batch Prediction.
Pilih model yang ingin dieksekusi (bisa salah satu atau keduanya).
Klik Predict batch → preview hasil dan tombol unduhan f1_batch_predictions.csv.
Kolom yang tidak ditemukan akan diisi nilai default dari slider. Pastikan header bersih untuk menghindari pergeseran tipe data.

📥 Skema Input Fitur
Sector & Straight-Line Pace
Fitur	Rentang	Default	Catatan
SpeedI1_mean	250–360 km/h	305.0	Rata-rata kecepatan sektor 1
SpeedI1_max	260–370 km/h	320.0	Puncak kecepatan sektor 1
SpeedI2_mean	250–360 km/h	300.0	Rata-rata sektor 2
SpeedI2_max	260–370 km/h	318.0	Puncak sektor 2
SpeedFL_mean	260–370 km/h	315.0	Rata-rata lap tercepat
SpeedFL_max	270–380 km/h	330.0	Puncak lap tercepat
SpeedST_mean	260–370 km/h	320.0	Rata-rata straights
SpeedST_max	270–380 km/h	336.0	Puncak straights
Race Consistency
Fitur	Rentang	Default	Catatan
Position_mean	1–20	8	Rata-rata posisi
Position_std	0–8	2.0	Deviasi posisi
Position_min	1–20	4	Posisi terbaik
Position_max	1–20	12	Posisi terburuk
lap_time_cv	0–0.15	0.04	Koefisien variasi lap time
position_changes	-15–15	1	Perubahan posisi neto
position_volatility	0–8	1.5	Rolling std perubahan posisi
Strategy & Tyre Management
Fitur	Rentang	Default	Catatan
TyreLife_mean	0–40 lap	18.0	Rata-rata umur ban
TyreLife_max	0–60 lap	30.0	Umur ban terpanjang
pit_stops	0–6	2	Jumlah pit-stop
fastest_lap_percentage	0–100%	12	Persentase lap <105% FL
top_10_laps_percentage	0–100%	45	Lap yang tembus top-10
Relative Pace Advantage
Fitur	Rentang	Default	Catatan
SpeedI1_mean_advantage	-15–15 km/h	0.5	Delta sektor 1 dibanding grid
SpeedI2_mean_advantage	-15–15 km/h	0.3	Delta sektor 2
SpeedFL_mean_advantage	-15–15 km/h	1.0	Delta rata-rata lap tercepat
SpeedST_mean_advantage	-15–15 km/h	0.8	Delta top speed lintasan lurus
Semua kolom bersifat numerik. Untuk batch CSV, gunakan titik desimal (.) dan hindari satuan teks.

🧠 Model Terlatih
Model	Artefak	Tugas	Metrik Utama
Finishing Position (SVR)	models/f1_position_model_svr.pkl	Regresi posisi	CV MAE 1.45 · Test MAE 3.65 · Test RMSE 4.89 · R² 0.60
Scored Points (Extra Trees)	models/f1_points_model_extra trees.pkl	Regresi poin	CV MAE 2.07 · Test MAE 5.38 · Test RMSE 6.92 · R² 0.41
Parameter pipeline, daftar fitur yang dipakai, dan metrik lengkap tersedia di models/f1_models_metadata.json. UI Model Cards membaca file ini secara langsung sehingga kamu cukup memperbarui JSON ketika mengganti artefak.

🔧 Menambah / Mengganti Model
Latih model baru dan simpan menggunakan joblib.dump.
Letakkan artefak .pkl di folder models/.
Perbarui models/f1_models_metadata.json:
filename menunjuk ke artefak relatif terhadap folder ini.
features_used harus sinkron dengan kolom CSV.
Tambahkan metrik baru pada bagian performance bila perlu.
Reload aplikasi Streamlit → model otomatis terdeteksi via metadata.
📓 Eksperimen
Notebook notebooks/nb-1.ipynb berisi eksperimen awal (EDA, feature engineering, pelatihan).
Simpan cache/data pendukung di notebooks/cache/ untuk menjaga repo utama tetap ringan.
Pertimbangkan untuk memindahkan pipeline produksi ke modul terpisah jika training makin kompleks.
🧭 Roadmap Saran
Tambah validasi schema (pydantic / pandera) sebelum inference batch.
Sematkan grafik distribusi input agar analis cepat melihat outlier.
Integrasi FastF1 API langsung di sidebar untuk menarik telemetry terbaru.
Otomatiskan retraining & update metadata dengan skrip CLI.
Sertakan license + panduan kontribusi agar proyek mudah dibuka untuk kolaborator.
