# Sistema de Detección de Anomalías - Dashboard

Dashboard web para monitoreo en tiempo real de anomalías en variables de proceso industrial.

## 🚀 Despliegue

Esta aplicación está lista para desplegarse en [Streamlit Cloud](https://streamlit.io/cloud).

## 📋 Características

- **Detección en tiempo real**: Monitoreo continuo de anomalías en variables de proceso
- **Vista operativa**: Diseñada para operarios de planta
- **Interfaz industrial**: Estilo Schneider Electric
- **Múltiples variables**: Análisis de todas las variables de proceso simultáneamente
- **Métricas del modelo**: MAE, RMSE, R², tasa de anomalías

## 🛠️ Tecnologías

- **Streamlit**: Framework web
- **Prophet**: Modelo de Machine Learning para series temporales
- **Plotly**: Visualizaciones interactivas
- **Pandas**: Procesamiento de datos

## 📁 Estructura

```
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias
├── .streamlit/
│   └── config.toml          # Configuración de tema
└── README.md                # Este archivo
```

## 🔧 Configuración Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

## 📊 Datos de Entrada

El dashboard espera archivos CSV con los siguientes campos:

- `ds` - Fecha/hora
- `y` - Valor real observado
- `yhat` - Valor predicho
- `yhat_lower` - Límite inferior del intervalo
- `yhat_upper` - Límite superior del intervalo
- `residual` - Diferencia entre real y predicho
- `is_anomaly` - Boolean: es anomalía
- `anomaly_score` - Score 0-100
- `variable` - Nombre de la variable
- `prediction_error_pct` - Error porcentual

## 📝 Notas

- El dashboard busca automáticamente el archivo más reciente de anomalías en `pipeline/results/`
- En producción, modificar `load_anomalies_data()` para conectar con tu fuente de datos
- Los modelos Prophet se cargan desde `pipeline/models/prophet/`
