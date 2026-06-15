# Customer Intelligence — Online Retail II

Proyecto de Data Science end-to-end sobre el dataset público **Online Retail II (UCI)**. Cubre desde la limpieza de datos hasta un modelo predictivo de churn con interpretabilidad SHAP, pasando por segmentación RFM avanzada, modelos probabilísticos de CLV y un dashboard interactivo desplegable en producción.

---

## Qué problema resuelve

Las empresas de retail acumulan millones de transacciones pero rara vez saben con precisión quiénes son sus mejores clientes, cuánto valen a futuro y cuáles están a punto de irse. Este proyecto construye un sistema completo de inteligencia de cliente que responde tres preguntas de negocio concretas:

- ¿Qué tipo de cliente es cada uno y qué comportamiento tiene?
- ¿Cuánto revenue generará en los próximos 12 meses?
- ¿Cuál es su probabilidad de abandonar y cuánto merece invertirse en retenerlo?

---

## Dataset

**Online Retail II** — UCI Machine Learning Repository  
802.949 transacciones · 5.862 clientes · Diciembre 2009 – Diciembre 2011  
Retail de regalos y artículos del hogar con sede en Reino Unido, ventas mayoritariamente B2B.

---

## Stack tecnológico

Python 3.11 · pandas · scikit-learn · XGBoost · lifetimes · SHAP · Plotly · Streamlit

---

## Estructura del proyecto

```
customer-decision-intelligence/
│
├── data/
│   ├── raw/                        # Dataset original UCI
│   └── processed/                  # Parquets limpios y enriquecidos
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb     # Carga y exploración inicial
│   ├── 02_eda.ipynb                # Análisis exploratorio completo
│   ├── 03_rfm_segmentation.ipynb   # RFM · K-Means · BG/NBD · CLV
│   ├── 04_dashboard.ipynb          # Visualizaciones Plotly interactivas
│   └── 05_churn_prediction.ipynb   # XGBoost · SHAP · Segmentación de riesgo
│
├── app/
│   └── streamlit_app.py            # Dashboard interactivo con carga de datos
│
├── outputs/
│   ├── 03_rfm/                     # Modelos BG/NBD, K-Means y gráficos
│   ├── 04_dashboard/               # HTMLs standalone exportados
│   └── 05_churn/                   # Modelo XGBoost, SHAP plots, predicciones
│
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 · Ingesta de datos
Carga del Excel original, detección de anomalías, primeros estadísticos descriptivos y decisiones de limpieza documentadas.

### 02 · EDA
Análisis exploratorio completo: distribuciones de revenue, estacionalidad, productos más vendidos, geografía de clientes y patrones de comportamiento de compra.

### 03 · Segmentación RFM avanzada
Cálculo de métricas Recency, Frequency y Monetary a nivel cliente. Clustering K-Means con selección de K por Elbow, Silhouette y Davies-Bouldin. Modelo probabilístico BG/NBD para estimar compras futuras y probabilidad de actividad. Modelo Gamma-Gamma para CLV a 12 meses. Tabla `customer_360.parquet` con perfil completo de cada cliente.

### 04 · Dashboard interactivo
Visualizaciones Plotly embebidas en el notebook: evolución temporal de revenue por segmento, scatter 3D del espacio RFM, mapas de calor, curva de Pareto de CLV y ranking de top clientes. Exportación como HTML standalone sin dependencias.

### 05 · Predicción de churn
Pipeline completo de clasificación binaria. Baseline con Logistic Regression, modelo principal XGBoost con `scale_pos_weight` para manejar desbalanceo y umbral optimizado por F1. Interpretabilidad con SHAP: beeswarm global, dependence plots y waterfall por cliente individual. Segmentación en cuatro niveles de riesgo cruzada con CLV para priorización de campañas de retención.

---

## App Streamlit

Dashboard interactivo con cinco secciones: evolución temporal, segmentos RFM, espacio 3D, CLV y P_alive, y ranking de top clientes. Permite cargar cualquier dataset propio en formato parquet o CSV y recalcula RFM y K-Means en tiempo real. Filtros por segmento en el sidebar y descarga de resultados en CSV.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Resultados clave

El modelo de churn predice con alta fiabilidad qué clientes van a abandonar antes de que lo hagan. Combinado con el CLV estimado por el modelo Gamma-Gamma, convierte una lista de scores en una herramienta de priorización de negocio: quién vale la pena retener y cuánto invertir en ello.

La segmentación K-Means identifica perfiles de cliente diferenciados con comportamientos de compra distintos, cada uno con sus métricas RFM características y su contribución al revenue total. El análisis de Pareto confirma la concentración de valor típica del retail: un porcentaje pequeño de clientes genera la mayoría del CLV futuro.

---

## Instalación

```bash
git clone https://github.com/tu-usuario/customer-decision-intelligence.git
cd customer-decision-intelligence

python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

Los notebooks se ejecutan en orden del 01 al 05. El dataset original debe descargarse de [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) y colocarse en `data/raw/`.

---

## Autor

**Esthe** · [LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/tu-usuario)
