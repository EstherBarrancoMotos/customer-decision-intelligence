"""
Customer Intelligence Dashboard — Streamlit App
================================================
Ejecutar: streamlit run app/streamlit_app.py
Requiere: pip install streamlit plotly pandas pyarrow lifetimes scikit-learn joblib
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json, joblib, io

# ── Lifetimes (opcional, para predicciones en vivo) ──────────────────────────
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    HAS_LIFETIMES = True
except ImportError:
    HAS_LIFETIMES = False

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title='Customer Intelligence Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

PALETTE = ['#2563EB','#7C3AED','#10B981','#F59E0B','#EF4444','#06B6D4','#D97706','#6B7280']

def hex_to_rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'


CLUSTER_LABELS_DEFAULT = {
    0: 'Champions',
    1: 'At Risk',
    2: 'Promising',
    3: 'Lost / Hibernating',
}

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background: #F1F5F9; }

    /* Header */
    .dash-header {
        background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 28px;
    }
    .dash-header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
    .dash-header p  { opacity: 0.85; font-size: 0.92rem; }

    /* KPI cards */
    .kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        flex: 1;
        min-width: 130px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
        text-align: center;
    }
    .kpi-icon  { font-size: 1.6rem; }
    .kpi-value { font-size: 1.45rem; font-weight: 700; }
    .kpi-label { font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: .05em; }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,.07);
        margin-bottom: 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1E293B !important; }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_parquet(path: Path):
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_parquet_bytes(data: bytes):
    return pd.read_parquet(io.BytesIO(data))

@st.cache_data(show_spinner=False)
def load_csv_bytes(data: bytes):
    return pd.read_csv(io.BytesIO(data))


def compute_rfm(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Calcula RFM desde un dataframe de transacciones."""
    df_tx = df_tx.copy()
    df_tx['InvoiceDate'] = pd.to_datetime(df_tx['InvoiceDate'])
    df_tx['Revenue']     = df_tx['Quantity'] * df_tx['Price']
    df_tx = df_tx[df_tx['Revenue'] > 0]

    snapshot = df_tx['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = (
        df_tx.groupby('Customer ID')
        .agg(
            last_purchase   = ('InvoiceDate', 'max'),
            frequency       = ('Invoice', 'nunique'),
            monetary        = ('Revenue', 'sum'),
            avg_order_value = ('Revenue', lambda x: x.groupby(
                df_tx.loc[x.index, 'Invoice']).sum().mean()),
        )
        .reset_index()
    )
    rfm['recency'] = (snapshot - rfm['last_purchase']).dt.days
    rfm = rfm.drop(columns='last_purchase')
    rfm = rfm[rfm['monetary'] > 0]

    # Scores
    rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'),  q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_total'] = rfm[['R_score','F_score','M_score']].sum(axis=1)

    def seg(row):
        r, f = row['R_score'], row['F_score']
        if r >= 4 and f >= 4:   return 'Champions'
        elif r >= 3 and f >= 3: return 'Loyal'
        elif r >= 4 and f <= 2: return 'New Customers'
        elif r >= 3 and f <= 2: return 'Potential Loyalists'
        elif r == 2:            return 'At Risk'
        elif r == 1 and f >= 3: return "Can't Lose Them"
        else:                   return 'Lost'

    rfm['RFM_segment'] = rfm.apply(seg, axis=1)
    return rfm, snapshot


def run_kmeans(rfm: pd.DataFrame, k: int) -> pd.DataFrame:
    """Ajusta K-Means y añade columna cluster_label."""
    features = ['recency', 'frequency', 'monetary', 'avg_order_value']
    X = rfm[features].copy()
    for col in ['frequency', 'monetary', 'avg_order_value']:
        X[col] = np.log1p(X[col])
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=42)
    rfm = rfm.copy()
    rfm['cluster'] = km.fit_predict(Xs)
    # Etiquetas automáticas basadas en monetary mediana
    med = rfm.groupby('cluster')['monetary'].median().sort_values(ascending=False)
    label_map = {c: f'Seg {i+1}' for i, c in enumerate(med.index)}
    rfm['cluster_label'] = rfm['cluster'].map(label_map)
    return rfm, km, scaler, Xs


def run_bgnbd(df_tx: pd.DataFrame, snapshot, penalizer=0.001):
    """Ajusta BG/NBD y Gamma-Gamma, devuelve summary con predicciones."""
    if not HAS_LIFETIMES:
        return None
    df_lt = df_tx.copy()
    df_lt['InvoiceDate'] = pd.to_datetime(df_lt['InvoiceDate'])
    df_lt['Revenue'] = df_lt['Quantity'] * df_lt['Price']
    df_lt = df_lt[df_lt['Revenue'] > 0]
    df_lt = df_lt.groupby(['Customer ID','Invoice','InvoiceDate'])['Revenue'].sum().reset_index()

    summary = summary_data_from_transaction_data(
        df_lt,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='Revenue',
        observation_period_end=snapshot,
        freq='D'
    )

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'], verbose=False)

    summary['p_alive'] = bgf.conditional_probability_alive(
        summary['frequency'], summary['recency'], summary['T']
    )
    for w in [4, 12, 26, 52]:
        summary[f'pred_{w}w'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            w, summary['frequency'], summary['recency'], summary['T']
        )

    summary_gg = summary[(summary['frequency'] > 0) & (summary['monetary_value'] > 0)].copy()
    if len(summary_gg) > 10:
        ggf = GammaGammaFitter(penalizer_coef=penalizer)
        ggf.fit(summary_gg['frequency'], summary_gg['monetary_value'], verbose=False)
        summary_gg['clv_12m'] = ggf.customer_lifetime_value(
            bgf, summary_gg['frequency'], summary_gg['recency'],
            summary_gg['T'], summary_gg['monetary_value'],
            time=12, freq='W', discount_rate=0.01
        )
        summary = summary.join(summary_gg[['clv_12m']], how='left')

    return summary


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 Customer Intelligence")
    st.markdown("---")

    st.markdown("### 📁 Fuente de datos")
    data_source = st.radio(
        "Selecciona la fuente:",
        ["Datos del proyecto (parquet)", "Subir archivo propio"],
        label_visibility='collapsed'
    )

    uploaded_c360 = None
    uploaded_tx   = None

    if data_source == "Subir archivo propio":
        st.markdown("**customer_360.parquet** o CSV:")
        uploaded_c360 = st.file_uploader(
            "Tabla 360 (generada en NB03)", type=['parquet','csv'], key='c360'
        )
        st.markdown("**transactions_clean.parquet** o CSV:")
        uploaded_tx = st.file_uploader(
            "Transacciones originales", type=['parquet','csv'], key='tx'
        )
        st.info("💡 Si solo subes transacciones, el dashboard calculará RFM y K-Means en tiempo real.")

    st.markdown("---")
    st.markdown("### ⚙️ Opciones")
    k_clusters = st.slider("Número de clusters K-Means", 2, 8, 4)
    show_bgnbd = st.checkbox("Calcular BG/NBD en vivo", value=False,
                              help="Requiere librería lifetimes. Puede tardar ~30s.")
    top_n      = st.slider("Top clientes por CLV", 10, 50, 25)

    st.markdown("---")
    st.markdown("### 🎨 Filtros")
    segment_filter = None  # se llena después de cargar datos


# =============================================================================
# CARGA DE DATOS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR   = PROJECT_ROOT / 'outputs' / '03_rfm'

c360 = None
tx   = None
rfm_computed = False

with st.spinner("Cargando datos..."):
    # ── Caso 1: datos del proyecto ────────────────────────────────────────────
    if data_source == "Datos del proyecto (parquet)":
        c360_path = DATA_DIR / 'customer_360.parquet'
        tx_path   = DATA_DIR / 'transactions_clean.parquet'

        if c360_path.exists():
            c360 = load_parquet(c360_path)
        if tx_path.exists():
            tx = load_parquet(tx_path)
            tx['InvoiceDate'] = pd.to_datetime(tx['InvoiceDate'])
            tx['Revenue']     = tx['Quantity'] * tx['Price']
            tx['YearMonth']   = tx['InvoiceDate'].dt.to_period('M').astype(str)

    # ── Caso 2: archivo subido ────────────────────────────────────────────────
    else:
        if uploaded_c360 is not None:
            raw = uploaded_c360.read()
            if uploaded_c360.name.endswith('.csv'):
                c360 = load_csv_bytes(raw)
            else:
                c360 = load_parquet_bytes(raw)

        if uploaded_tx is not None:
            raw = uploaded_tx.read()
            if uploaded_tx.name.endswith('.csv'):
                tx = load_csv_bytes(raw)
            else:
                tx = load_parquet_bytes(raw)

            tx['InvoiceDate'] = pd.to_datetime(tx['InvoiceDate'])
            tx['Revenue']     = tx['Quantity'] * tx['Price']
            tx['YearMonth']   = tx['InvoiceDate'].dt.to_period('M').astype(str)

    # ── Si hay transacciones pero no c360, calculamos RFM + KMeans en vivo ───
    if c360 is None and tx is not None:
        with st.spinner("Calculando RFM y K-Means desde transacciones..."):
            rfm_raw, snapshot_date = compute_rfm(tx)
            c360, km_live, scaler_live, Xs_live = run_kmeans(rfm_raw, k_clusters)

            if show_bgnbd and HAS_LIFETIMES:
                summary_live = run_bgnbd(tx, snapshot_date)
                if summary_live is not None:
                    c360 = c360.set_index('Customer ID').join(
                        summary_live[['p_alive','clv_12m'] + [f'pred_{w}w' for w in [4,12,26,52]]],
                        how='left'
                    ).reset_index()
            rfm_computed = True

    # ── Merge tx + c360 para evolución temporal ───────────────────────────────
    if tx is not None and c360 is not None and 'cluster_label' in c360.columns:
        tx = tx.merge(
            c360[['Customer ID','cluster_label','RFM_segment']],
            on='Customer ID', how='left'
        )


# =============================================================================
# VALIDACIÓN
# =============================================================================
if c360 is None:
    st.markdown("""
    <div class="dash-header">
        <h1>📊 Customer Intelligence Dashboard</h1>
        <p>Segmentación RFM · K-Means · BG/NBD · CLV</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ No se encontraron datos. Selecciona una fuente en el sidebar.")
    st.markdown("""
    **Opciones:**
    - **Datos del proyecto**: asegúrate de haber ejecutado el Notebook 03 y que `data/processed/customer_360.parquet` existe.
    - **Subir archivo propio**: sube `customer_360.parquet` o `transactions_clean.parquet` (o CSV equivalente).

    **Columnas necesarias en transactions:**
    `Customer ID`, `Invoice`, `InvoiceDate`, `Quantity`, `Price`
    """)
    st.stop()


# =============================================================================
# FILTROS SIDEBAR (después de cargar datos)
# =============================================================================
with st.sidebar:
    if 'cluster_label' in c360.columns:
        all_segs = sorted(c360['cluster_label'].dropna().unique().tolist())
        segment_filter = st.multiselect(
            "Filtrar segmentos:",
            options=all_segs,
            default=all_segs,
            key='seg_filter'
        )

# Aplicar filtro
c360_f = c360.copy()
if segment_filter and 'cluster_label' in c360_f.columns:
    c360_f = c360_f[c360_f['cluster_label'].isin(segment_filter)]


# =============================================================================
# HEADER
# =============================================================================
period_start = tx['InvoiceDate'].min().strftime('%b %Y') if tx is not None else '—'
period_end   = tx['InvoiceDate'].max().strftime('%b %Y') if tx is not None else '—'

st.markdown(f"""
<div class="dash-header">
    <h1>Customer Intelligence Dashboard</h1>
    <p>Segmentacion RFM · K-Means · BG/NBD · CLV | Online Retail II (UCI) | {period_start} - {period_end}{' | RFM calculado en vivo' if rfm_computed else ''}</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# KPIs
# =============================================================================
total_revenue   = tx['Revenue'].sum() if tx is not None else c360_f['monetary'].sum()
total_customers = c360_f['Customer ID'].nunique() if 'Customer ID' in c360_f.columns else len(c360_f)
total_orders    = tx['Invoice'].nunique() if tx is not None else 0
avg_basket      = c360_f['avg_order_value'].median() if 'avg_order_value' in c360_f.columns else 0
total_clv       = c360_f['clv_12m'].sum() if 'clv_12m' in c360_f.columns else 0
pct_alive       = (c360_f['p_alive'] >= 0.5).mean()*100 if 'p_alive' in c360_f.columns else 0
champions_n     = (c360_f['RFM_segment'] == 'Champions').sum() if 'RFM_segment' in c360_f.columns else 0
at_risk_n       = (c360_f['RFM_segment'] == 'At Risk').sum()   if 'RFM_segment' in c360_f.columns else 0

cols = st.columns(8)
kpi_data = [
    ('💰', 'Revenue',    f'£{total_revenue:,.0f}',  '#2563EB'),
    ('👥', 'Clientes',   f'{total_customers:,}',      '#7C3AED'),
    ('🛒', 'Órdenes',    f'{total_orders:,}',          '#10B981'),
    ('📦', 'AOV Med.',   f'£{avg_basket:,.0f}',       '#F59E0B'),
    ('🔮', 'CLV 12m',   f'£{total_clv:,.0f}',        '#2563EB'),
    ('✅', 'P_alive≥50%',f'{pct_alive:.1f}%',          '#10B981'),
    ('⭐', 'Champions',  f'{champions_n:,}',            '#F59E0B'),
    ('⚠️', 'En Riesgo', f'{at_risk_n:,}',             '#EF4444'),
]
for col, (icon, label, value, color) in zip(cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")  # spacer


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Evolución Temporal",
    "🎯 Segmentos RFM",
    "🌐 Espacio 3D",
    "💰 CLV & P_alive",
    "🏆 Top Clientes"
])


# ── TAB 1: Evolución temporal ─────────────────────────────────────────────────
with tab1:
    if tx is not None:
        monthly = (
            tx.groupby('YearMonth')
            .agg(revenue=('Revenue','sum'), orders=('Invoice','nunique'),
                 customers=('Customer ID','nunique'))
            .reset_index().sort_values('YearMonth')
        )

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Revenue Mensual (£)', 'Órdenes por Mes', 'Clientes Activos'],
            horizontal_spacing=0.06
        )
        for i, (col, color) in enumerate(
            [('revenue','#2563EB'),('orders','#7C3AED'),('customers','#10B981')]
        ):
            fig.add_trace(go.Scatter(
                x=monthly['YearMonth'], y=monthly[col],
                mode='lines+markers',
                line=dict(color=color, width=2.5), marker=dict(size=5),
                fill='tozeroy', fillcolor=hex_to_rgba(color, 0.12), showlegend=False
            ), row=1, col=i+1)

        fig.update_layout(height=360, hovermode='x unified', margin=dict(t=50,b=30))
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        # Revenue por segmento
        if 'cluster_label' in tx.columns:
            monthly_seg = (
                tx.dropna(subset=['cluster_label'])
                .groupby(['YearMonth','cluster_label'])['Revenue'].sum()
                .reset_index().sort_values('YearMonth')
            )
            fig2 = px.area(
                monthly_seg, x='YearMonth', y='Revenue', color='cluster_label',
                color_discrete_sequence=PALETTE,
                labels={'Revenue':'Revenue (£)','YearMonth':'Mes','cluster_label':'Segmento'},
                title='Revenue Mensual por Segmento'
            )
            fig2.update_layout(height=380, hovermode='x unified')
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sube el archivo de transacciones para ver la evolución temporal.")


# ── TAB 2: Segmentos RFM ──────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        # Sunburst
        if 'RFM_segment' in c360_f.columns and 'cluster_label' in c360_f.columns:
            sb_df = (
                c360_f.groupby(['RFM_segment','cluster_label'])
                .agg(n=('Customer ID' if 'Customer ID' in c360_f.columns else c360_f.columns[0],'count'),
                     revenue=('monetary','sum'))
                .reset_index()
            )
            fig = px.sunburst(
                sb_df, path=['RFM_segment','cluster_label'], values='n',
                color='revenue', color_continuous_scale='Blues',
                title='Distribución: RFM → Cluster'
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Heatmap RFM
        if 'R_score' in c360_f.columns and 'F_score' in c360_f.columns:
            hm = (
                c360_f.groupby(['R_score','F_score'])['monetary'].median()
                .reset_index()
                .pivot(index='R_score', columns='F_score', values='monetary')
                .sort_index(ascending=False)
            )
            fig = go.Figure(go.Heatmap(
                z=hm.values,
                x=[f'F={c}' for c in hm.columns],
                y=[f'R={r}' for r in hm.index],
                colorscale='Blues',
                text=np.round(hm.values,0).astype(int),
                texttemplate='£%{text}', textfont={'size':11}
            ))
            fig.update_layout(
                title='Mapa de Calor RFM — Monetary Mediano',
                height=450,
                xaxis_title='Frequency Score',
                yaxis_title='Recency Score'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Violin plots
    if 'cluster_label' in c360_f.columns:
        fig = make_subplots(1, 3, subplot_titles=['Recency','log(Frequency)','log(Monetary)'])
        for ci, (metric, log_s) in enumerate([('recency',False),('frequency',True),('monetary',True)]):
            for i, seg in enumerate(sorted(c360_f['cluster_label'].dropna().unique())):
                data = c360_f.loc[c360_f['cluster_label']==seg, metric]
                if log_s: data = np.log1p(data)
                fig.add_trace(go.Violin(
                    y=data, name=seg,
                    line_color=PALETTE[i%len(PALETTE)],
                    fillcolor=PALETTE[i%len(PALETTE)],
                    opacity=0.7, box_visible=True, meanline_visible=True,
                    showlegend=(ci==0)
                ), row=1, col=ci+1)
        fig.update_layout(title='Distribución RFM por Segmento', height=420, violinmode='group')
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 3: Espacio 3D ─────────────────────────────────────────────────────────
with tab3:
    if 'cluster_label' in c360_f.columns:
        plot3d = c360_f.copy()
        plot3d['monetary_log']  = np.log1p(plot3d['monetary'])
        plot3d['frequency_log'] = np.log1p(plot3d['frequency'])
        q99 = plot3d['monetary_log'].quantile(0.99)
        plot3d = plot3d[plot3d['monetary_log'] <= q99]

        hover_cols = {
            'recency': True, 'frequency': True, 'monetary': ':.0f',
            'monetary_log': False, 'frequency_log': False
        }
        if 'RFM_segment' in plot3d.columns:
            hover_cols['RFM_segment'] = True
        if 'Customer ID' in plot3d.columns:
            hover_cols['Customer ID'] = True

        fig = px.scatter_3d(
            plot3d, x='recency', y='frequency_log', z='monetary_log',
            color='cluster_label', color_discrete_sequence=PALETTE,
            size='avg_order_value' if 'avg_order_value' in plot3d.columns else None,
            size_max=12, opacity=0.55,
            hover_data=hover_cols,
            labels={
                'recency':'Recency (días)',
                'frequency_log':'log(Frequency)',
                'monetary_log':'log(Monetary £)',
                'cluster_label':'Segmento'
            },
            title='Espacio RFM 3D por Segmento'
        )
        fig.update_layout(height=600, scene=dict(bgcolor='#F8FAFC'))
        st.plotly_chart(fig, use_container_width=True)

        # PCA 2D
        st.markdown("##### Proyección PCA 2D")
        features = ['recency','frequency','monetary']
        if 'avg_order_value' in c360_f.columns:
            features.append('avg_order_value')
        X_pca_in = c360_f[features].copy()
        for col in features[1:]:
            X_pca_in[col] = np.log1p(X_pca_in[col])
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(RobustScaler().fit_transform(X_pca_in))
        pca_df = c360_f[['cluster_label']].copy()
        pca_df['PC1'] = Xp[:,0]
        pca_df['PC2'] = Xp[:,1]
        var = pca.explained_variance_ratio_

        fig2 = px.scatter(
            pca_df, x='PC1', y='PC2', color='cluster_label',
            color_discrete_sequence=PALETTE, opacity=0.5,
            labels={'cluster_label':'Segmento',
                    'PC1':f'PC1 ({var[0]*100:.1f}%)',
                    'PC2':f'PC2 ({var[1]*100:.1f}%)'},
            title=f'PCA 2D — Varianza explicada: {sum(var)*100:.1f}%'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)


# ── TAB 4: CLV & P_alive ──────────────────────────────────────────────────────
with tab4:
    has_clv    = 'clv_12m'  in c360_f.columns
    has_palive = 'p_alive'  in c360_f.columns

    if not has_clv and not has_palive:
        st.info("ℹ️ No hay columnas CLV / P_alive. Activa 'Calcular BG/NBD en vivo' o sube customer_360.parquet del notebook 03.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if has_palive:
                fig = px.histogram(
                    c360_f.dropna(subset=['p_alive']),
                    x='p_alive', color='cluster_label' if 'cluster_label' in c360_f.columns else None,
                    color_discrete_sequence=PALETTE, nbins=40, barmode='overlay', opacity=0.7,
                    labels={'p_alive':'P(alive)','cluster_label':'Segmento'},
                    title='Distribución P_alive por Segmento'
                )
                fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                              annotation_text='Umbral 50%')
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if has_clv and 'cluster_label' in c360_f.columns:
                plot_box = c360_f.dropna(subset=['clv_12m']).copy()
                q98 = plot_box['clv_12m'].quantile(0.98)
                plot_box = plot_box[plot_box['clv_12m'] <= q98]
                fig = px.box(
                    plot_box, x='cluster_label', y='clv_12m',
                    color='cluster_label', color_discrete_sequence=PALETTE,
                    points='outliers',
                    labels={'clv_12m':'CLV 12 meses (£)','cluster_label':'Segmento'},
                    title='CLV 12 meses por Segmento'
                )
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Scatter P_alive vs CLV
        if has_clv and has_palive:
            plot_sc = c360_f.dropna(subset=['clv_12m','p_alive']).copy()
            q98 = plot_sc['clv_12m'].quantile(0.98)
            plot_sc = plot_sc[plot_sc['clv_12m'] <= q98]

            hover = {'p_alive':':.3f','clv_12m':':.0f'}
            if 'Customer ID' in plot_sc.columns:
                hover['Customer ID'] = True
            if 'frequency' in plot_sc.columns:
                hover['frequency'] = True
            if 'RFM_segment' in plot_sc.columns:
                hover['RFM_segment'] = True

            fig = px.scatter(
                plot_sc, x='p_alive', y='clv_12m',
                color='cluster_label' if 'cluster_label' in plot_sc.columns else None,
                size='frequency' if 'frequency' in plot_sc.columns else None,
                color_discrete_sequence=PALETTE, opacity=0.55, size_max=16,
                hover_data=hover,
                labels={'p_alive':'P(alive)','clv_12m':'CLV 12m (£)','cluster_label':'Segmento'},
                title='P_alive vs CLV 12 meses — Mapa de Valor del Cliente'
            )
            fig.add_vline(x=0.5, line_dash='dot', line_color='gray', opacity=0.4)
            if 'clv_12m' in plot_sc.columns:
                fig.add_hline(y=plot_sc['clv_12m'].median(), line_dash='dot',
                              line_color='gray', opacity=0.4, annotation_text='CLV mediano')
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

        # Curva Pareto
        if has_clv:
            pareto = (
                c360_f.dropna(subset=['clv_12m'])
                .sort_values('clv_12m', ascending=False).copy()
            )
            pareto['cum_pct']      = pareto['clv_12m'].cumsum() / pareto['clv_12m'].sum() * 100
            pareto['customer_pct'] = np.arange(1, len(pareto)+1) / len(pareto) * 100

            fig = go.Figure(go.Scatter(
                x=pareto['customer_pct'], y=pareto['cum_pct'],
                mode='lines', line=dict(color='#2563EB', width=3),
                fill='tozeroy', fillcolor='rgba(37,99,235,0.1)', name='CLV acumulado'
            ))
            fig.add_hline(y=80, line_dash='dash', line_color='#EF4444',
                          annotation_text='80% del CLV')
            idx80  = (pareto['cum_pct'] >= 80).idxmax()
            pct80  = pareto.loc[idx80, 'customer_pct']
            fig.add_vline(x=pct80, line_dash='dot', line_color='#F59E0B',
                          annotation_text=f'{pct80:.1f}% clientes')
            fig.update_layout(
                title=f'Curva de Pareto CLV — el {pct80:.1f}% de clientes genera el 80% del CLV',
                xaxis_title='% Clientes', yaxis_title='% CLV acumulado', height=400
            )
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 5: Top Clientes ───────────────────────────────────────────────────────
with tab5:
    sort_col = 'clv_12m' if 'clv_12m' in c360_f.columns else 'monetary'
    sort_label = 'CLV 12m' if sort_col == 'clv_12m' else 'Monetary'

    display_cols = ['Customer ID'] if 'Customer ID' in c360_f.columns else []
    for c in ['cluster_label','RFM_segment','recency','frequency','monetary',
              'avg_order_value','p_alive','clv_12m']:
        if c in c360_f.columns:
            display_cols.append(c)

    top_df = (
        c360_f.dropna(subset=[sort_col])
        .nlargest(top_n, sort_col)
        [display_cols]
        .reset_index(drop=True)
    )
    top_df.index += 1

    # Bar chart
    id_col = 'Customer ID' if 'Customer ID' in top_df.columns else top_df.columns[0]
    fig = go.Figure(go.Bar(
        x=top_df[sort_col],
        y=top_df[id_col].astype(str),
        orientation='h',
        marker=dict(
            color=top_df[sort_col],
            colorscale='Blues', showscale=True,
            colorbar=dict(title=f'{sort_label} £')
        ),
        text=[f'£{v:,.0f}' for v in top_df[sort_col]],
        textposition='outside'
    ))
    fig.update_layout(
        title=f'Top {top_n} Clientes por {sort_label}',
        xaxis_title=f'{sort_label} (£)',
        yaxis_title='Customer ID',
        height=max(400, top_n * 22),
        yaxis={'categoryorder':'total ascending'},
        margin=dict(l=100, r=160)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla interactiva
    st.markdown("##### 📋 Tabla detallada")
    st.dataframe(
        top_df.style.background_gradient(subset=[sort_col], cmap='Blues'),
        use_container_width=True,
        height=400
    )

    # Descarga CSV
    csv = top_df.to_csv(index=True).encode('utf-8')
    st.download_button(
        label=f"⬇️ Descargar Top {top_n} como CSV",
        data=csv,
        file_name=f'top{top_n}_clientes.csv',
        mime='text/csv'
    )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<center style='color:#94A3B8;font-size:0.8rem'>"
    "Customer Intelligence Portfolio · Online Retail II (UCI) · "
    "Python · Streamlit · Plotly · lifetimes · scikit-learn"
    "</center>",
    unsafe_allow_html=True
)
