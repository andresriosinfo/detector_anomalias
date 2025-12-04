"""
Dashboard - Sistema de Detección de Anomalías
Vista operativa para monitoreo en tiempo real de anomalías en variables de proceso
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Detección de Anomalías",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado con fuente Nunito y estilo Schneider
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Nunito', -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Helvetica, Arial, sans-serif;
        color: #333333;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #2E9A42;
    }
    
    .stApp {
        background-color: #F5F5F5;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def load_anomalies_data():
    """
    Carga los datos de anomalías detectadas.
    
    NOTA: En producción, reemplazar con la salida directa del detector.
    """
    try:
        # Intentar cargar desde diferentes ubicaciones posibles
        paths = [
            'pipeline/results/anomalies_detected_20251202_124025.csv',
            'pipeline/results/anomalies_detected_*.csv',
            'results/anomalies_detected_*.csv',
            'anomalies_detected.csv'
        ]
        
        df = None
        from pathlib import Path
        import glob
        
        # Buscar archivos de anomalías
        for pattern in [
            'pipeline/results/anomalies_detected_*.csv',
            'results/anomalies_detected_*.csv',
            'anomalies_detected*.csv'
        ]:
            files = glob.glob(pattern)
            if files:
                # Tomar el más reciente
                latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                df = pd.read_csv(latest_file)
                break
        
        if df is None:
            st.warning("No se encontró archivo de anomalías. Usando datos de ejemplo.")
            # Crear datos de ejemplo
            dates = pd.date_range(start='2025-01-01', periods=1000, freq='10min')
            variables = ['ARO_FC103', 'ARO_LC101', 'ARO_PC101', 'ARO_TC105']
            df = pd.DataFrame({
                'ds': np.repeat(dates, len(variables)),
                'variable': np.tile(variables, len(dates)),
                'y': np.random.normal(100, 10, len(dates) * len(variables)),
                'yhat': np.random.normal(100, 8, len(dates) * len(variables)),
                'yhat_lower': np.random.normal(95, 8, len(dates) * len(variables)),
                'yhat_upper': np.random.normal(105, 8, len(dates) * len(variables)),
                'residual': np.random.normal(0, 2, len(dates) * len(variables)),
                'is_anomaly': np.random.choice([0, 1], size=len(dates) * len(variables), p=[0.95, 0.05]),
                'anomaly_score': np.random.uniform(0, 100, len(dates) * len(variables)),
                'prediction_error_pct': np.random.uniform(0, 10, len(dates) * len(variables))
            })
            df['outside_interval'] = (df['y'] < df['yhat_lower']) | (df['y'] > df['yhat_upper'])
            df['high_residual'] = np.abs(df['residual']) > 3
            df['is_anomaly'] = (df['outside_interval'] | df['high_residual']).astype(int)
        
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


def load_model_metrics():
    """Carga las métricas del modelo."""
    try:
        from pathlib import Path
        import glob
        
        files = glob.glob('pipeline/results/model_metrics_*.csv')
        if files:
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
            return pd.read_csv(latest_file)
        return None
    except:
        return None


def get_current_status(df):
    """Obtiene el estado actual del sistema (últimas anomalías)."""
    if len(df) == 0:
        return None
    
    # Último timestamp
    ultimo_ts = df['ds'].max()
    
    # Anomalías en las últimas 2 horas
    cutoff = ultimo_ts - timedelta(hours=2)
    df_recent = df[df['ds'] >= cutoff]
    
    n_anomalies = df_recent['is_anomaly'].sum()
    n_total = len(df_recent)
    tasa = (n_anomalies / n_total * 100) if n_total > 0 else 0
    
    # Variables afectadas
    variables_afectadas = df_recent[df_recent['is_anomaly'] == 1]['variable'].nunique()
    
    # Anomalía más reciente
    ultima_anomalia = df[df['is_anomaly'] == 1].iloc[-1] if (df['is_anomaly'] == 1).any() else None
    
    return {
        'ultimo_timestamp': ultimo_ts,
        'anomalias_2h': n_anomalies,
        'total_puntos_2h': n_total,
        'tasa_anomalias': tasa,
        'variables_afectadas': variables_afectadas,
        'ultima_anomalia': ultima_anomalia,
        'hay_anomalias_activas': n_anomalies > 0
    }


def plot_anomaly_trend(df, variable=None, n_points=200):
    """Gráfico de tendencia con anomalías."""
    # Filtrar por variable si se especifica
    if variable:
        df_plot = df[df['variable'] == variable].tail(n_points).copy()
    else:
        # Si no hay variable, tomar las últimas N muestras de todas las variables
        df_plot = df.tail(n_points).copy()
    
    if len(df_plot) == 0:
        return None
    
    # Si hay múltiples variables, agrupar por timestamp
    if variable is None:
        df_plot = df_plot.groupby('ds').agg({
            'y': 'mean',
            'yhat': 'mean',
            'yhat_lower': 'mean',
            'yhat_upper': 'mean',
            'is_anomaly': 'sum'
        }).reset_index()
    
    fig = go.Figure()
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat_upper'],
        mode='lines',
        name='Límite Superior',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat_lower'],
        mode='lines',
        name='Intervalo de Confianza',
        fill='tonexty',
        fillcolor='rgba(61, 205, 88, 0.1)',
        line=dict(width=0),
        showlegend=True
    ))
    
    # Valor predicho
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat'],
        mode='lines',
        name='Valor Predicho',
        line=dict(color='#2E9A42', width=2, dash='dot'),
        hovertemplate='%{x}<br>Predicho: %{y:.2f}<extra></extra>'
    ))
    
    # Valor real
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['y'],
        mode='lines',
        name='Valor Real',
        line=dict(color='#3DCD58', width=2),
        hovertemplate='%{x}<br>Real: %{y:.2f}<extra></extra>'
    ))
    
    # Anomalías
    if variable:
        anomalias = df_plot[df_plot['is_anomaly'] == 1] if 'is_anomaly' in df_plot.columns else pd.DataFrame()
    else:
        # Para múltiples variables, marcar timestamps con anomalías
        anomalias = df_plot[df_plot['is_anomaly'] > 0] if 'is_anomaly' in df_plot.columns else pd.DataFrame()
    
    if len(anomalias) > 0:
        fig.add_trace(go.Scatter(
            x=anomalias['ds'],
            y=anomalias['y'],
            mode='markers',
            name='Anomalías',
            marker=dict(
                color='#DC143C',
                size=10,
                symbol='x',
                line=dict(width=2, color='white')
            ),
            hovertemplate='%{x}<br>Anomalía: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='Tiempo',
        yaxis_title='Valor',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=20, b=50),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def main():
    """Función principal de la aplicación."""
    
    # Cargar datos
    df = load_anomalies_data()
    if df is None:
        st.stop()
    
    # Cargar métricas del modelo
    df_metrics = load_model_metrics()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuración")
        
        # Selector de variable
        variables = sorted(df['variable'].unique()) if 'variable' in df.columns else []
        variable_selected = st.selectbox(
            "Variable a visualizar",
            options=['Todas'] + variables,
            index=0
        )
        
        horas_visualizar = st.slider(
            "Horas a visualizar",
            min_value=1,
            max_value=48,
            value=24,
            step=1
        )
        
        min_score = st.slider(
            "Score mínimo de anomalía",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
    
    # Obtener estado actual
    estado = get_current_status(df)
    
    if estado is None:
        st.error("No hay datos disponibles")
        st.stop()
    
    # ==========================================
    # SECCIÓN PRINCIPAL: ESTADO ACTUAL
    # ==========================================
    
    st.title("Sistema de Detección de Anomalías")
    st.markdown("---")
    
    # Tarjeta de estado principal
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if estado['hay_anomalias_activas']:
            st.markdown("""
            <div style='background-color: #DC143C; color: white; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 2rem;'>ANOMALÍAS DETECTADAS</h2>
                <p style='font-size: 1rem; margin-top: 0.5rem;'>En las últimas 2 horas</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #3DCD58; color: white; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 2rem;'>SISTEMA NORMAL</h2>
                <p style='font-size: 1rem; margin-top: 0.5rem;'>Sin anomalías recientes</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 2px solid #E5E5E5; text-align: center;'>
            <div style='color: #333333; font-size: 0.9rem; margin-bottom: 0.5rem;'>ANOMALÍAS (2h)</div>
            <div style='color: #DC143C; font-size: 3rem; font-weight: 700;'>{estado['anomalias_2h']}</div>
            <div style='color: #666666; font-size: 0.8rem; margin-top: 0.5rem;'>Tasa: {estado['tasa_anomalias']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 2px solid #E5E5E5; text-align: center;'>
            <div style='color: #333333; font-size: 0.9rem; margin-bottom: 0.5rem;'>VARIABLES AFECTADAS</div>
            <div style='color: #2E9A42; font-size: 3rem; font-weight: 700;'>{estado['variables_afectadas']}</div>
            <div style='color: #666666; font-size: 0.8rem; margin-top: 0.5rem;'>De {df['variable'].nunique()} totales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ultima_ts = estado['ultimo_timestamp']
        st.markdown(f"""
        <div style='background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 2px solid #E5E5E5; text-align: center;'>
            <div style='color: #333333; font-size: 0.9rem; margin-bottom: 0.5rem;'>ÚLTIMA ACTUALIZACIÓN</div>
            <div style='color: #333333; font-size: 1.5rem; font-weight: 600; margin-top: 0.5rem;'>
                {ultima_ts.strftime('%H:%M:%S')}
            </div>
            <div style='color: #666666; font-size: 0.8rem; margin-top: 0.5rem;'>
                {ultima_ts.strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==========================================
    # GRÁFICO PRINCIPAL
    # ==========================================
    st.markdown("## Tendencias y Anomalías")
    
    var_plot = None if variable_selected == 'Todas' else variable_selected
    n_points = int(horas_visualizar * 6)  # Asumiendo datos cada 10 min
    
    fig_trend = plot_anomaly_trend(df, variable=var_plot, n_points=n_points)
    if fig_trend:
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================
    # TOP VARIABLES CON MÁS ANOMALÍAS
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Variables con Más Anomalías (Últimas 24h)")
        
        # Filtrar últimas 24h
        cutoff_24h = df['ds'].max() - timedelta(hours=24)
        df_24h = df[df['ds'] >= cutoff_24h]
        
        # Resumen por variable
        summary = df_24h.groupby('variable').agg({
            'is_anomaly': ['sum', 'count'],
            'anomaly_score': 'mean'
        }).reset_index()
        summary.columns = ['variable', 'n_anomalias', 'n_total', 'score_promedio']
        summary['tasa'] = (summary['n_anomalias'] / summary['n_total'] * 100).round(2)
        summary = summary.sort_values('n_anomalias', ascending=False).head(10)
        
        # Gráfico de barras
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=summary['variable'],
            y=summary['n_anomalias'],
            marker_color='#DC143C',
            text=summary['n_anomalias'],
            textposition='outside',
            hovertemplate='%{x}<br>Anomalías: %{y}<extra></extra>'
        ))
        
        fig_bars.update_layout(
            xaxis_title='Variable',
            yaxis_title='Número de Anomalías',
            height=300,
            template='plotly_white',
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        
        st.plotly_chart(fig_bars, use_container_width=True)
    
    with col2:
        st.markdown("### Distribución de Anomalías")
        
        # Distribución por score
        df_anomalies = df[df['is_anomaly'] == 1].copy()
        if len(df_anomalies) > 0:
            # Categorías de score
            df_anomalies['categoria'] = pd.cut(
                df_anomalies['anomaly_score'],
                bins=[0, 50, 75, 100],
                labels=['Bajo (0-50)', 'Medio (50-75)', 'Alto (75-100)']
            )
            
            distrib = df_anomalies['categoria'].value_counts()
            
            # Gráfico donut
            fig_donut = go.Figure(data=[go.Pie(
                labels=distrib.index,
                values=distrib.values,
                hole=0.5,
                marker_colors=['#3DCD58', '#FFA500', '#DC143C'],
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_donut.update_layout(
                height=300,
                template='plotly_white',
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                showlegend=False
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No hay anomalías para mostrar")
    
    st.markdown("---")
    
    # ==========================================
    # TABLA DE ANOMALÍAS RECIENTES
    # ==========================================
    st.markdown("### Anomalías Recientes")
    
    # Filtrar anomalías con score mínimo
    df_anomalies_recent = df[
        (df['is_anomaly'] == 1) & 
        (df['anomaly_score'] >= min_score)
    ].sort_values('ds', ascending=False).head(50)
    
    if len(df_anomalies_recent) > 0:
        # Seleccionar columnas relevantes
        cols_display = ['ds', 'variable', 'y', 'yhat', 'anomaly_score', 'prediction_error_pct']
        cols_available = [c for c in cols_display if c in df_anomalies_recent.columns]
        
        df_display = df_anomalies_recent[cols_available].copy()
        df_display['ds'] = df_display['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Formatear números
        if 'y' in df_display.columns:
            df_display['y'] = df_display['y'].round(2)
        if 'yhat' in df_display.columns:
            df_display['yhat'] = df_display['yhat'].round(2)
        if 'anomaly_score' in df_display.columns:
            df_display['anomaly_score'] = df_display['anomaly_score'].round(1)
        if 'prediction_error_pct' in df_display.columns:
            df_display['prediction_error_pct'] = df_display['prediction_error_pct'].round(2)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info(f"No hay anomalías con score >= {min_score}")
    
    st.markdown("---")
    
    # ==========================================
    # MÉTRICAS DEL MODELO (colapsable)
    # ==========================================
    if df_metrics is not None:
        with st.expander("Métricas del Modelo por Variable"):
            # Mostrar métricas principales
            cols_metrics = ['variable', 'mae', 'rmse', 'r2', 'anomaly_rate_pct', 'avg_anomaly_score']
            cols_available = [c for c in cols_metrics if c in df_metrics.columns]
            
            if cols_available:
                df_metrics_display = df_metrics[cols_available].copy()
                # Formatear números
                for col in df_metrics_display.select_dtypes(include=[np.number]).columns:
                    df_metrics_display[col] = df_metrics_display[col].round(3)
                
                st.dataframe(df_metrics_display, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; padding: 1rem; font-size: 0.85rem;'>
        Sistema de Detección de Anomalías - Modelo Prophet | Monitoreo en Tiempo Real
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

