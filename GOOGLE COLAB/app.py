import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Credit Card Fraud Dashboard', layout='wide')

DATA_PATH = 'creditcard_synthetic.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()
normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

st.title('Credit Card Fraud Analytics Dashboard')
st.caption('Dashboard interaktif untuk analisis transaksi normal dan fraud')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total transaksi', f'{len(df):,}')
col2.metric('Normal', f'{len(normal):,}')
col3.metric('Fraud', f'{len(fraud):,}')
col4.metric('Rasio fraud', f'{df["Class"].mean() * 100:.2f}%')

feature_diffs = []
for col in [f'V{i}' for i in range(1, 29)]:
    diff = abs(fraud[col].mean() - normal[col].mean())
    feature_diffs.append((col, diff))

# Top 5 features
feature_rows = sorted(feature_diffs, key=lambda x: x[1], reverse=True)[:5]
feature_names = [row[0] for row in feature_rows]
feature_values = [row[1] for row in feature_rows]

st.subheader('Ringkasan utama')
left, right = st.columns([1.3, 1])

with left:
    fig_pie = px.pie(
        names=['Normal', 'Fraud'],
        values=[len(normal), len(fraud)],
        color_discrete_sequence=['#2ecc71', '#e74c3c'],
        hole=0.45,
        title='Distribusi transaksi per kelas'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    fig_bar = px.bar(
        x=feature_names,
        y=feature_values,
        labels={'x': 'Fitur', 'y': 'Selisih rata-rata'},
        color=feature_values,
        color_continuous_scale='Blues',
        title='5 fitur paling berbeda'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader('Analisis distribusi dan pola')
col_a, col_b = st.columns(2)

with col_a:
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=normal['Amount'], name='Normal', marker_color='#2ecc71'))
    fig_box.add_trace(go.Box(y=fraud['Amount'], name='Fraud', marker_color='#e74c3c'))
    fig_box.update_layout(
        title='Distribusi amount per kelas',
        xaxis_title='Kelas',
        yaxis_title='Amount',
        template='plotly_white'
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_b:
    scatter_x = st.selectbox('Pilih fitur X untuk scatter', [f'V{i}' for i in range(1, 29)], index=9)
    scatter_y = st.selectbox('Pilih fitur Y untuk scatter', [f'V{i}' for i in range(1, 29)], index=17)
    fig_scatter = px.scatter(
        df,
        x=scatter_x,
        y=scatter_y,
        color='Class',
        color_continuous_scale='RdYlGn_r',
        title=f'{scatter_x} vs {scatter_y}',
        labels={'Class': 'Label'}
    )
    fig_scatter.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader('Insight utama')
st.markdown(
    """
    - Dataset sangat tidak seimbang: hanya 0,5% transaksi masuk kategori fraud.
    - Rata-rata amount pada transaksi fraud jauh lebih tinggi dibanding normal.
    - Fitur V10, V18, V15, V8, dan V25 menunjukkan perbedaan paling besar antar kelas.
    - Pola fitur ini berpotensi menjadi indikator utama untuk deteksi transaksi penipuan.
    """
)

st.sidebar.header('Pengaturan dashboard')
st.sidebar.write('Pilih parameter tampilan:')
threshold = st.sidebar.slider('Ambang batas amount untuk highlight', 0, 2000, 500, 50)
filtered = df[df['Amount'] >= threshold]

st.sidebar.subheader('Data yang difilter')
st.sidebar.write(f'Jumlah transaksi di atas ambang batas: {len(filtered):,}')

st.sidebar.dataframe(filtered.head(10), use_container_width=True)
