import pandas as pd
import numpy as np

# Set seed untuk reproduksibilitas
np.random.seed(42)

# Jumlah sampel total dan rasio penipuan (imbalanced dataset)
n_samples = 10000
fraud_ratio = 0.005  # 0.5% transaksi penipuan
n_fraud = int(n_samples * fraud_ratio)
n_normal = n_samples - n_fraud

# Generate fitur Time (sekensial dalam detik)
time = np.sort(np.random.uniform(0, 172800, n_samples))

# Generate fitur ter-PCA V1 hingga V28
v_features_normal = np.random.normal(loc=0, scale=1, size=(n_normal, 28))
v_features_fraud = np.random.normal(loc=1.5, scale=2, size=(n_fraud, 28))
v_features = np.vstack((v_features_normal, v_features_fraud))

# Generate fitur Amount (nominal transaksi)
amount_normal = np.random.exponential(scale=88, size=n_normal)
amount_fraud = np.random.uniform(low=100, high=2000, size=n_fraud)
amount = np.concatenate((amount_normal, amount_fraud))

# Target variabel Class (0: Normal, 1: Fraud)
labels = np.concatenate((np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)))

# Gabungkan semua variabel menjadi DataFrame
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
data = np.column_stack((time, v_features, amount, labels))
df = pd.DataFrame(data, columns=columns)
df['Class'] = df['Class'].astype(int)

# Acak urutan baris data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan dataset ke berkas CSV lokal
output_file = 'creditcard_synthetic.csv'
df.to_csv(output_file, index=False)
print(f"Dataset sintetik berhasil dibuat: {output_file} ({len(df)} baris)")