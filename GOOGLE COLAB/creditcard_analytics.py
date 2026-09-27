import pandas as pd

DATA_PATH = 'creditcard_synthetic.csv'
SUMMARY_PATH = 'creditcard_analytics_summary.csv'
FEATURE_PATH = 'creditcard_feature_differences.csv'


def main():
    df = pd.read_csv(DATA_PATH)

    normal = df[df['Class'] == 0]
    fraud = df[df['Class'] == 1]

    summary = {
        'total_rows': int(len(df)),
        'normal_rows': int(len(normal)),
        'fraud_rows': int(len(fraud)),
        'fraud_rate_pct': round(df['Class'].mean() * 100, 4),
        'amount_mean_all': round(df['Amount'].mean(), 2),
        'amount_mean_normal': round(normal['Amount'].mean(), 2),
        'amount_mean_fraud': round(fraud['Amount'].mean(), 2),
        'amount_median_fraud': round(fraud['Amount'].median(), 2),
        'amount_max_fraud': round(fraud['Amount'].max(), 2),
        'time_mean_normal': round(normal['Time'].mean(), 2),
        'time_mean_fraud': round(fraud['Time'].mean(), 2),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    feature_rows = []
    for col in [f'V{i}' for i in range(1, 29)]:
        normal_mean = normal[col].mean()
        fraud_mean = fraud[col].mean()
        diff = fraud_mean - normal_mean
        feature_rows.append({
            'feature': col,
            'normal_mean': round(normal_mean, 4),
            'fraud_mean': round(fraud_mean, 4),
            'mean_difference': round(diff, 4),
            'abs_mean_difference': round(abs(diff), 4),
        })

    feature_df = pd.DataFrame(feature_rows).sort_values('abs_mean_difference', ascending=False)
    feature_df.to_csv(FEATURE_PATH, index=False)

    print('=== Ringkasan Dataset ===')
    print(summary_df.to_string(index=False))
    print('\n=== 10 Fitur dengan Perbedaan Terbesar ===')
    print(feature_df.head(10).to_string(index=False))
    print(f'\nFile ringkasan disimpan ke: {SUMMARY_PATH}')
    print(f'File fitur disimpan ke: {FEATURE_PATH}')


if __name__ == '__main__':
    main()
