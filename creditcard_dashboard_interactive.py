import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH = 'creditcard_synthetic.csv'
OUTPUT_PATH = 'creditcard_dashboard_interactive.html'


def main():
    df = pd.read_csv(DATA_PATH)
    normal = df[df['Class'] == 0]
    fraud = df[df['Class'] == 1]

    fraud_rate = round(df['Class'].mean() * 100, 2)

    feature_diffs = []
    for col in [f'V{i}' for i in range(1, 29)]:
        diff = abs(fraud[col].mean() - normal[col].mean())
        feature_diffs.append((col, diff))
    top_features = sorted(feature_diffs, key=lambda x: x[1], reverse=True)[:5]
    feature_names = [item[0] for item in top_features]
    feature_values = [item[1] for item in top_features]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            'Distribusi Kelas',
            'Amount per Kelas',
            'Pola V10 vs V18',
            '5 Fitur Paling Berbeda'
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.2
    )

    fig.add_trace(
        go.Pie(
            labels=['Normal', 'Fraud'],
            values=[len(normal), len(fraud)],
            hole=0.35,
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<extra></extra>'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Box(
            y=normal['Amount'],
            name='Normal',
            marker_color='#2ecc71',
            boxmean=True,
            hovertemplate='Normal<br>Amount: %{y}<extra></extra>'
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Box(
            y=fraud['Amount'],
            name='Fraud',
            marker_color='#e74c3c',
            boxmean=True,
            hovertemplate='Fraud<br>Amount: %{y}<extra></extra>'
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Scatter(
            x=normal['V10'],
            y=normal['V18'],
            mode='markers',
            name='Normal',
            marker=dict(color='#2ecc71', size=7, opacity=0.6),
            hovertemplate='Normal<br>V10: %{x}<br>V18: %{y}<extra></extra>'
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=fraud['V10'],
            y=fraud['V18'],
            mode='markers',
            name='Fraud',
            marker=dict(color='#e74c3c', size=8, opacity=0.8),
            hovertemplate='Fraud<br>V10: %{x}<br>V18: %{y}<extra></extra>'
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=feature_values,
            y=feature_names,
            orientation='h',
            marker=dict(color='#3498db'),
            hovertemplate='%{y}<br>Selisih rata-rata: %{x:.4f}<extra></extra>'
        ),
        row=2,
        col=2
    )

    fig.update_xaxes(title_text='Jumlah', row=1, col=1)
    fig.update_xaxes(title_text='Amount', row=1, col=2)
    fig.update_xaxes(title_text='V10', row=2, col=1)
    fig.update_xaxes(title_text='Nilai Selisih', row=2, col=2)
    fig.update_yaxes(title_text='V18', row=2, col=1)
    fig.update_yaxes(title_text='Fitur', row=2, col=2)

    fig.update_layout(
        title='Dashboard Analitik Fraud Credit Card',
        title_x=0.5,
        template='plotly_white',
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=30, r=30, t=70, b=30)
    )

    fig.add_annotation(
        text=(
            f'Jumlah total: {len(df)}<br>'
            f'Rasio fraud: {fraud_rate}%<br>'
            f'Avg amount normal: {normal["Amount"].mean():.2f}<br>'
            f'Avg amount fraud: {fraud["Amount"].mean():.2f}'
        ),
        align='left',
        showarrow=False,
        x=1.02,
        y=0.05,
        xref='paper',
        yref='paper',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='lightgray'
    )

    fig.write_html(OUTPUT_PATH, auto_open=False)
    print(f'Dashboard interaktif dibuat: {OUTPUT_PATH}')
    print(f'Rasio fraud: {fraud_rate}%')
    print('Top feature indicators:', feature_names)


if __name__ == '__main__':
    main()
