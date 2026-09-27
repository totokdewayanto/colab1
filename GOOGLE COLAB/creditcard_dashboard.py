import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = 'creditcard_synthetic.csv'
DASHBOARD_PATH = 'creditcard_dashboard.png'


def main():
    df = pd.read_csv(DATA_PATH)
    normal = df[df['Class'] == 0]
    fraud = df[df['Class'] == 1]

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Credit Card Fraud Dashboard', fontsize=18, fontweight='bold')

    # 1. Distribution of classes
    labels = ['Normal', 'Fraud']
    counts = [len(normal), len(fraud)]
    axes[0, 0].bar(labels, counts, color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Distribusi Kelas')
    axes[0, 0].set_ylabel('Jumlah Transaksi')
    for bar, value in zip(axes[0, 0].patches, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha='center', va='bottom')

    # 2. Amount comparison between classes
    box = axes[0, 1].boxplot([normal['Amount'], fraud['Amount']], patch_artist=True)
    for patch in box['boxes']:
        patch.set(facecolor='#5dade2', alpha=0.8)
    for median in box['medians']:
        median.set(color='#1f618d')
    axes[0, 1].set_xticklabels(['Normal', 'Fraud'])
    axes[0, 1].set_title('Distribusi Amount per Kelas')
    axes[0, 1].set_ylabel('Amount')

    # 3. Top feature differences
    feature_diffs = []
    for col in [f'V{i}' for i in range(1, 29)]:
        diff = abs(fraud[col].mean() - normal[col].mean())
        feature_diffs.append((col, diff))
    top_features = sorted(feature_diffs, key=lambda x: x[1], reverse=True)[:5]
    feature_names = [item[0] for item in top_features]
    feature_values = [item[1] for item in top_features]
    axes[1, 0].barh(feature_names[::-1], feature_values[::-1], color='#3498db')
    axes[1, 0].set_title('5 Fitur dengan Perbedaan Tertinggi')
    axes[1, 0].set_xlabel('|Rata-rata Fraud - Normal|')

    # 4. Time distribution overlay
    axes[1, 1].hist(normal['Time'], bins=30, alpha=0.7, color='#2ecc71', label='Normal')
    axes[1, 1].hist(fraud['Time'], bins=30, alpha=0.7, color='#e74c3c', label='Fraud')
    axes[1, 1].set_title('Distribusi Waktu')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].legend()

    # Summary text
    summary_text = (
        f'Jumlah Total: {len(df)}\n'
        f'Rasio Fraud: {df["Class"].mean() * 100:.2f}%\n'
        f'Avg Amount Normal: {normal["Amount"].mean():.2f}\n'
        f'Avg Amount Fraud: {fraud["Amount"].mean():.2f}'
    )
    fig.text(0.72, 0.12, summary_text, fontsize=10, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(DASHBOARD_PATH, dpi=200)
    plt.close(fig)

    print(f'Dashboard created: {DASHBOARD_PATH}')
    print('Fraud rate:', round(df['Class'].mean() * 100, 2), '%')
    print('Top features:', feature_names)


if __name__ == '__main__':
    main()
