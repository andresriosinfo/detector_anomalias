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
    page_icon=None,
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
        else:
            df = None
        
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
        df = df.sort_values(['variable', 'ds']).reset_index(drop=True)
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


def detect_persistent_anomalies(df, threshold_hours=1):
    """
    Detecta anomalías que persisten por más de threshold_hours.
    Retorna DataFrame con variables que requieren revisión.
    """
    df_anomalies = df[df['is_anomaly'] == 1].copy()
    
    if len(df_anomalies) == 0:
        return pd.DataFrame()
    
    persistent_vars = []
    
    # Agrupar por variable
    for variable in df_anomalies['variable'].unique():
        df_var = df_anomalies[df_anomalies['variable'] == variable].sort_values('ds')
        
        # Encontrar grupos consecutivos de anomalías
        df_var['group'] = (df_var['ds'].diff() > timedelta(hours=threshold_hours)).cumsum()
        
        # Calcular duración de cada grupo
        for group_id in df_var['group'].unique():
            group = df_var[df_var['group'] == group_id]
            duration = (group['ds'].max() - group['ds'].min()).total_seconds() / 3600
            
            if duration >= threshold_hours:
                persistent_vars.append({
                    'variable': variable,
                    'inicio': group['ds'].min(),
                    'fin': group['ds'].max(),
                    'duracion_horas': duration,
                    'n_anomalias': len(group),
                    'score_promedio': group['anomaly_score'].mean(),
                    'score_maximo': group['anomaly_score'].max()
                })
    
    return pd.DataFrame(persistent_vars)


def get_variables_to_review(df):
    """
    Identifica variables que requieren revisión basado en:
    1. Anomalías persistentes (1 hora)
    2. Variables con más anomalías recientes (últimas 24h)
    """
    ultimo_ts = df['ds'].max()
    cutoff_24h = ultimo_ts - timedelta(hours=24)
    df_recent = df[df['ds'] >= cutoff_24h]
    
    # 1. Anomalías persistentes
    persistent = detect_persistent_anomalies(df_recent, threshold_hours=1)
    
    # 2. Variables con más anomalías recientes
    anomalies_by_var = df_recent[df_recent['is_anomaly'] == 1].groupby('variable').agg({
        'is_anomaly': 'count',
        'anomaly_score': ['mean', 'max']
    }).reset_index()
    anomalies_by_var.columns = ['variable', 'n_anomalias', 'score_promedio', 'score_maximo']
    anomalies_by_var = anomalies_by_var.sort_values('n_anomalias', ascending=False)
    
    # Combinar información
    review_vars = []
    
    # Variables con anomalías persistentes (prioridad alta)
    if len(persistent) > 0:
        for _, row in persistent.iterrows():
            review_vars.append({
                'variable': row['variable'],
                'razon': 'Anomalía persistente',
                'prioridad': 'ALTA',
                'detalle': f"Persiste desde {row['inicio'].strftime('%H:%M')} ({row['duracion_horas']:.1f}h)",
                'n_anomalias': row['n_anomalias'],
                'score_maximo': row['score_maximo']
            })
    
    # Variables con muchas anomalías recientes (prioridad media)
    top_vars = anomalies_by_var.head(10)
    for _, row in top_vars.iterrows():
        # Solo agregar si no está ya en la lista por persistencia
        if row['variable'] not in [v['variable'] for v in review_vars]:
            review_vars.append({
                'variable': row['variable'],
                'razon': 'Múltiples anomalías',
                'prioridad': 'MEDIA',
                'detalle': f"{row['n_anomalias']} anomalías en últimas 24h",
                'n_anomalias': row['n_anomalias'],
                'score_maximo': row['score_maximo']
            })
    
    return pd.DataFrame(review_vars)


def plot_anomaly_trend(df, variable, show_title=True):
    """
    Gráfico de tendencia con anomalías - Última hora.
    El intervalo de confianza está alrededor de la predicción (yhat).
    Muestra cómo se detectan las anomalías: valores fuera del intervalo de confianza.
    """
    if variable is None or variable == 'Todas':
        return None
    
    # Filtrar última hora
    ultimo_ts = df['ds'].max()
    cutoff_1h = ultimo_ts - timedelta(hours=1)
    df_plot = df[(df['variable'] == variable) & (df['ds'] >= cutoff_1h)].copy()
    
    if len(df_plot) == 0:
        return None
    
    # Ordenar por fecha
    df_plot = df_plot.sort_values('ds').reset_index(drop=True)
    
    fig = go.Figure()
    
    # IMPORTANTE: El intervalo de confianza está centrado alrededor de yhat (predicción)
    # yhat_lower < yhat < yhat_upper
    # El área sombreada debe mostrar el intervalo alrededor de yhat
    
    # 1. Primero dibujamos el límite SUPERIOR (invisible, solo para crear el área)
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat_upper'],
        mode='lines',
        name='_upper_invisible',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 2. Luego el límite INFERIOR con fill='tonexty' (rellena hacia arriba hasta yhat_upper)
    # Esto crea el área sombreada del intervalo de confianza
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat_lower'],
        mode='lines',
        name='Intervalo de Confianza (95%)',
        fill='tonexty',  # Rellena hacia la traza anterior (yhat_upper)
        fillcolor='rgba(200, 200, 200, 0.2)',
        line=dict(width=0),
        showlegend=True,
        hovertemplate='%{x}<br>Intervalo: [%{y:.2f}, %{customdata:.2f}]<extra></extra>',
        customdata=df_plot['yhat_upper']
    ))
    
    # 3. Ahora dibujamos yhat (predicción) - debe estar VISUALMENTE en el centro del área sombreada
    # Esta línea va en el medio del intervalo yhat_lower < yhat < yhat_upper
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat'],
        mode='lines',
        name='Valor Predicho (yhat)',
        line=dict(color='#808080', width=3, dash='dot'),
        hovertemplate='%{x}<br>Predicho: %{y:.2f}<extra></extra>'
    ))
    
    # 4. Líneas de límites visibles (gris claro, punteadas) para ver los bordes del intervalo
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat_upper'],
        mode='lines',
        name='Límite Superior',
        line=dict(color='#B0B0B0', width=1.5, dash='dash'),
        opacity=0.8,
        showlegend=True,
        hovertemplate='%{x}<br>Límite Superior: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat_lower'],
        mode='lines',
        name='Límite Inferior',
        line=dict(color='#B0B0B0', width=1.5, dash='dash'),
        opacity=0.8,
        showlegend=True,
        hovertemplate='%{x}<br>Límite Inferior: %{y:.2f}<extra></extra>'
    ))
    
    # Valor real observado - azul claro
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['y'],
        mode='lines',
        name='Valor Real',
        line=dict(color='#87CEEB', width=2),
        hovertemplate='%{x}<br>Real: %{y:.2f}<extra></extra>'
    ))
    
    # Anomalías detectadas (valores reales fuera del intervalo de confianza) - rojo suave
    anomalias = df_plot[df_plot['is_anomaly'] == 1]
    
    if len(anomalias) > 0:
        # Separar anomalías por tipo: fuera del intervalo
        anomalias_arriba = anomalias[anomalias['y'] > anomalias['yhat_upper']]
        anomalias_abajo = anomalias[anomalias['y'] < anomalias['yhat_lower']]
        
        # Anomalías por encima del límite superior
        if len(anomalias_arriba) > 0:
            fig.add_trace(go.Scatter(
                x=anomalias_arriba['ds'],
                y=anomalias_arriba['y'],
                mode='markers',
                name='Anomalía',
                marker=dict(
                    color='#FF6B6B',
                    size=10,
                    symbol='triangle-up',
                    line=dict(width=1, color='#CC5555')
                ),
                hovertemplate='%{x}<br>Anomalía: %{y:.2f}<br>Límite: %{customdata:.2f}<br>Score: %{text:.1f}<extra></extra>',
                customdata=anomalias_arriba['yhat_upper'],
                text=anomalias_arriba['anomaly_score']
            ))
        
        # Anomalías por debajo del límite inferior
        if len(anomalias_abajo) > 0:
            fig.add_trace(go.Scatter(
                x=anomalias_abajo['ds'],
                y=anomalias_abajo['y'],
                mode='markers',
                name='Anomalía',
                marker=dict(
                    color='#FF6B6B',
                    size=10,
                    symbol='triangle-down',
                    line=dict(width=1, color='#CC5555')
                ),
                hovertemplate='%{x}<br>Anomalía: %{y:.2f}<br>Límite: %{customdata:.2f}<br>Score: %{text:.1f}<extra></extra>',
                customdata=anomalias_abajo['yhat_lower'],
                text=anomalias_abajo['anomaly_score'],
                showlegend=False
            ))
    
    # Estadísticas de anomalías para el título
    n_anomalies = len(anomalias) if len(anomalias) > 0 else 0
    tasa_anomalies = (n_anomalies / len(df_plot) * 100) if len(df_plot) > 0 else 0
    
    title_text = f'{variable}'
    if n_anomalies > 0:
        title_text += f' | {n_anomalies} anomalías ({tasa_anomalies:.1f}%)'
    
    fig.update_layout(
        title=title_text if show_title else None,
        xaxis_title='Tiempo',
        yaxis_title='Valor',
        hovermode='x unified',
        template='plotly_white',
        height=400 if not show_title else 450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=50, t=60 if show_title else 20, b=50),
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
    
    # Obtener variables que requieren revisión
    df_review = get_variables_to_review(df)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuración")
        
        # Selector de variable (priorizar las que requieren revisión)
        variables = sorted(df['variable'].unique()) if 'variable' in df.columns else []
        
        # Variables prioritarias primero
        if len(df_review) > 0:
            priority_vars = df_review['variable'].tolist()
            other_vars = [v for v in variables if v not in priority_vars]
            variables_sorted = priority_vars + other_vars
        else:
            variables_sorted = variables
        
        variable_selected = st.selectbox(
            "Variable a visualizar",
            options=['Todas'] + variables_sorted,
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
    
    # Estado actual
    ultimo_ts = df['ds'].max()
    cutoff_2h = ultimo_ts - timedelta(hours=2)
    df_recent_2h = df[df['ds'] >= cutoff_2h]
    n_anomalies_2h = df_recent_2h['is_anomaly'].sum()
    variables_afectadas = df_recent_2h[df_recent_2h['is_anomaly'] == 1]['variable'].nunique() if n_anomalies_2h > 0 else 0
    
    # ==========================================
    # SECCIÓN PRINCIPAL: ESTADO ACTUAL
    # ==========================================
    
    st.title("Sistema de Detección de Anomalías")
    st.markdown("---")
    
    # Alerta de anomalías persistentes
    if len(df_review) > 0:
        persistent_count = len(df_review[df_review['prioridad'] == 'ALTA'])
        if persistent_count > 0:
            st.markdown(f"""
            <div style='background-color: #DC143C; color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h3 style='color: white; margin: 0;'>ALERTA: {persistent_count} Variable(s) con Anomalías Persistentes (>1h)</h3>
                <p style='margin: 0.5rem 0 0 0;'>Revisar sección "Variables que Requieren Revisión"</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tarjetas de estado
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if n_anomalies_2h > 0:
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
        tasa = (n_anomalies_2h / len(df_recent_2h) * 100) if len(df_recent_2h) > 0 else 0
        st.markdown(f"""
        <div style='background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 2px solid #E5E5E5; text-align: center;'>
            <div style='color: #333333; font-size: 0.9rem; margin-bottom: 0.5rem;'>ANOMALÍAS (2h)</div>
            <div style='color: #DC143C; font-size: 3rem; font-weight: 700;'>{n_anomalies_2h}</div>
            <div style='color: #666666; font-size: 0.8rem; margin-top: 0.5rem;'>Tasa: {tasa:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 2px solid #E5E5E5; text-align: center;'>
            <div style='color: #333333; font-size: 0.9rem; margin-bottom: 0.5rem;'>VARIABLES AFECTADAS</div>
            <div style='color: #2E9A42; font-size: 3rem; font-weight: 700;'>{variables_afectadas}</div>
            <div style='color: #666666; font-size: 0.8rem; margin-top: 0.5rem;'>De {df['variable'].nunique()} totales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background-color: #FFFFFF; padding: 1.5rem; border-radius: 12px; border: 2px solid #E5E5E5; text-align: center;'>
            <div style='color: #333333; font-size: 0.9rem; margin-bottom: 0.5rem;'>ÚLTIMA ACTUALIZACIÓN</div>
            <div style='color: #333333; font-size: 1.5rem; font-weight: 600; margin-top: 0.5rem;'>
                {ultimo_ts.strftime('%H:%M:%S')}
            </div>
            <div style='color: #666666; font-size: 0.8rem; margin-top: 0.5rem;'>
                {ultimo_ts.strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==========================================
    # VARIABLES QUE REQUIEREN REVISIÓN
    # ==========================================
    if len(df_review) > 0:
        st.markdown("## Variables que Requieren Revisión")
        
        # Separar por prioridad
        alta_prioridad = df_review[df_review['prioridad'] == 'ALTA']
        media_prioridad = df_review[df_review['prioridad'] == 'MEDIA']
        
        if len(alta_prioridad) > 0:
            st.markdown("### Prioridad ALTA - Anomalías Persistentes (>1h)")
            for _, row in alta_prioridad.iterrows():
                st.markdown(f"""
                <div style='background-color: #FFE5E5; padding: 1rem; border-radius: 8px; border-left: 5px solid #DC143C; margin-bottom: 0.5rem;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong style='color: #DC143C; font-size: 1.1rem;'>{row['variable']}</strong>
                            <p style='margin: 0.3rem 0; color: #333333;'>{row['detalle']}</p>
                        </div>
                        <div style='text-align: right;'>
                            <div style='color: #666666; font-size: 0.9rem;'>{row['n_anomalias']} anomalías</div>
                            <div style='color: #DC143C; font-weight: 600;'>Score: {row['score_maximo']:.1f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if len(media_prioridad) > 0:
            st.markdown("### Prioridad MEDIA - Múltiples Anomalías Recientes")
            for _, row in media_prioridad.head(5).iterrows():
                st.markdown(f"""
                <div style='background-color: #FFF5E5; padding: 1rem; border-radius: 8px; border-left: 5px solid #FFA500; margin-bottom: 0.5rem;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong style='color: #FFA500; font-size: 1.1rem;'>{row['variable']}</strong>
                            <p style='margin: 0.3rem 0; color: #333333;'>{row['detalle']}</p>
                        </div>
                        <div style='text-align: right;'>
                            <div style='color: #666666; font-size: 0.9rem;'>{row['n_anomalias']} anomalías</div>
                            <div style='color: #FFA500; font-weight: 600;'>Score: {row['score_maximo']:.1f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ==========================================
    # SERIES TEMPORALES DE VARIABLES CON ANOMALÍAS
    # ==========================================
    if len(df_review) > 0:
        st.markdown("## Series Temporales de Variables con Anomalías")
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 1rem; border-radius: 8px; border-left: 4px solid #2E9A42; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #333333;'>
                <strong>Explicación de la detección:</strong> Las anomalías se detectan cuando el valor real (línea verde) 
                sale fuera del intervalo de confianza (área sombreada) alrededor de la predicción (línea punteada verde). 
                Los triángulos rojos indican las anomalías detectadas.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Obtener variables prioritarias para mostrar
        vars_to_plot = df_review['variable'].unique()[:6]  # Máximo 6 variables
        
        # Crear gráficos en columnas (2 columnas)
        n_cols = 2
        
        for i, var in enumerate(vars_to_plot):
            if i % n_cols == 0:
                cols = st.columns(n_cols)
            
            with cols[i % n_cols]:
                fig = plot_anomaly_trend(df, var, show_title=True)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    # ==========================================
    # GRÁFICO PRINCIPAL (solo si se selecciona una variable)
    # ==========================================
    if variable_selected != 'Todas':
        st.markdown(f"## Tendencias: {variable_selected}")
        st.markdown("*Última hora de datos*")
        
        fig_trend = plot_anomaly_trend(df, variable_selected, show_title=False)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
    
    # ==========================================
    # TOP VARIABLES CON MÁS ANOMALÍAS
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Variables con Más Anomalías (Últimas 24h)")
        
        cutoff_24h = df['ds'].max() - timedelta(hours=24)
        df_24h = df[df['ds'] >= cutoff_24h]
        
        summary = df_24h[df_24h['is_anomaly'] == 1].groupby('variable').agg({
            'is_anomaly': 'count',
            'anomaly_score': 'mean'
        }).reset_index()
        summary.columns = ['variable', 'n_anomalias', 'score_promedio']
        summary = summary.sort_values('n_anomalias', ascending=False).head(10)
        
        if len(summary) > 0:
            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(
                x=summary['variable'],
                y=summary['n_anomalias'],
                marker_color='#DC143C',
                text=summary['n_anomalias'],
                textposition='outside',
                hovertemplate='%{x}<br>Anomalías: %{y}<br>Score promedio: %{customdata:.1f}<extra></extra>',
                customdata=summary['score_promedio']
            ))
            
            fig_bars.update_layout(
                xaxis_title='Variable',
                yaxis_title='Número de Anomalías',
                height=350,
                template='plotly_white',
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF'
            )
            
            st.plotly_chart(fig_bars, use_container_width=True)
        else:
            st.info("No hay anomalías en las últimas 24 horas")
    
    with col2:
        st.markdown("### Distribución de Anomalías por Severidad")
        
        df_anomalies = df[df['is_anomaly'] == 1].copy()
        if len(df_anomalies) > 0:
            df_anomalies['categoria'] = pd.cut(
                df_anomalies['anomaly_score'],
                bins=[0, 50, 75, 100],
                labels=['Bajo (0-50)', 'Medio (50-75)', 'Alto (75-100)']
            )
            
            distrib = df_anomalies['categoria'].value_counts()
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=distrib.index,
                values=distrib.values,
                hole=0.5,
                marker_colors=['#3DCD58', '#FFA500', '#DC143C'],
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_donut.update_layout(
                height=350,
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
    
    df_anomalies_recent = df[
        (df['is_anomaly'] == 1) & 
        (df['anomaly_score'] >= min_score)
    ].sort_values('ds', ascending=False).head(50)
    
    if len(df_anomalies_recent) > 0:
        cols_display = ['ds', 'variable', 'y', 'yhat', 'anomaly_score', 'prediction_error_pct']
        cols_available = [c for c in cols_display if c in df_anomalies_recent.columns]
        
        df_display = df_anomalies_recent[cols_available].copy()
        df_display['ds'] = df_display['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Formatear números
        for col in ['y', 'yhat', 'anomaly_score', 'prediction_error_pct']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(2)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info(f"No hay anomalías con score >= {min_score}")
    
    st.markdown("---")
    
    # ==========================================
    # MÉTRICAS DEL MODELO (colapsable)
    # ==========================================
    if df_metrics is not None:
        with st.expander("Métricas del Modelo por Variable"):
            cols_metrics = ['variable', 'mae', 'rmse', 'r2', 'anomaly_rate_pct', 'avg_anomaly_score']
            cols_available = [c for c in cols_metrics if c in df_metrics.columns]
            
            if cols_available:
                df_metrics_display = df_metrics[cols_available].copy()
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
