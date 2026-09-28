# Clustering data pengukuran balita Triwulan II 2025
# Instalasi: pip install pandas openpyxl scikit-learn matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

FILE = "DATA PENGUKURAN BALITA TW. II 2025.xlsx"
SHEETS = ["April2025", "Mei2025", "Juni2025"]
FEATURES = ["ZS BB/U", "ZS TB/U", "ZS BB/TB"]

# Baca hanya kolom yang diperlukan. Header berada pada baris Excel ke-2.
parts = []
for sheet in SHEETS:
    df = pd.read_excel(
        FILE, sheet_name=sheet, header=1,
        usecols=["No", "Nama", "Tanggal Pengukuran"] + FEATURES
    )
    df["Bulan"] = sheet.replace("2025", "")
    parts.append(df)

data = pd.concat(parts, ignore_index=True)
for c in FEATURES:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Kode 999.99 dan nilai Z-score yang tidak wajar dikeluarkan.
valid = data[FEATURES].notna().all(axis=1)
valid &= data[FEATURES].ne(999.99).all(axis=1)
valid &= data[FEATURES].abs().le(10).all(axis=1)
clean = data.loc[valid].copy()
X = clean[FEATURES].astype(float)

# Standardisasi
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Evaluasi k=2 sampai 7 pada sampel acak agar lebih hemat waktu.
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(Xs), min(20000, len(Xs)), replace=False)
Xe = Xs[sample_idx]
metrics = []
for k in range(2, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Xe)
    metrics.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(Xe, labels, random_state=42),
        "davies_bouldin": davies_bouldin_score(Xe, labels)
    })
evaluation = pd.DataFrame(metrics)

# Pemilihan otomatis berdasarkan Silhouette tertinggi.
# Tinjau juga interpretasi substantif sebelum menetapkan k untuk publikasi.
best_k = int(evaluation.loc[evaluation.silhouette.idxmax(), "k"])
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clean["Cluster"] = final_model.fit_predict(Xs) + 1

profile = clean.groupby("Cluster")[FEATURES].agg(
    ["count", "mean", "median", "std"]
)
counts_month = pd.crosstab(clean["Bulan"], clean["Cluster"])

# PCA hanya untuk visualisasi; plot memakai sampel agar ringan.
pca = PCA(n_components=2)
pca_xy = pca.fit_transform(Xe)
# Label sampel dievaluasi ulang menggunakan model final
sample_labels = final_model.predict(Xe) + 1
plt.figure(figsize=(9, 6))
plt.scatter(pca_xy[:, 0], pca_xy[:, 1], c=sample_labels, s=5, alpha=.45)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("K-Means Clustering: PCA sample")
plt.tight_layout()
plt.savefig("visualisasi_pca_clustering.png", dpi=180)
plt.close()

# Simpan hasil tanpa nama balita untuk mengurangi paparan data pribadi.
result_export = clean[["Bulan", "Tanggal Pengukuran"] + FEATURES + ["Cluster"]]
with pd.ExcelWriter("hasil_clustering_balita.xlsx", engine="openpyxl") as writer:
    evaluation.to_excel(writer, index=False, sheet_name="Evaluasi_k")
    profile.to_excel(writer, sheet_name="Profil_Cluster")
    counts_month.to_excel(writer, sheet_name="Cluster_per_Bulan")
    result_export.to_excel(writer, index=False, sheet_name="Hasil_Sampel")

print("Total baris:", len(data))
print("Sampel valid:", len(clean))
print("Sampel dikeluarkan:", len(data)-len(clean))
print("Jumlah cluster terpilih:", best_k)
print(evaluation.round(4).to_string(index=False))
print(profile.round(3))
