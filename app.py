import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

# --- TRANSFORMADOR CUSTOMIZADO ---
# Necessário para carregar o pipeline do Sklearn com sucesso
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.limits = None

    def fit(self, X, y=None):
        if hasattr(X, 'quantile'): 
            self.limits = X.quantile(self.threshold)
        else: 
            self.limits = np.percentile(X, self.threshold * 100, axis=0)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if hasattr(X_copy, 'columns'):
            for i, col in enumerate(X_copy.columns):
                if self.limits is not None and col in self.limits:
                    limit = self.limits[col]
                    X_copy[col] = np.where(X_copy[col] > limit, limit, X_copy[col])
        elif self.limits is not None:
            for i in range(X_copy.shape[1]):
                limit = self.limits[i]
                X_copy[:, i] = np.where(X_copy[:, i] > limit, limit, X_copy[:, i])
        return X_copy

# --- CONFIGURAÇÃO E ESTILOS ---
st.set_page_config(
    page_title="Dashboard de Pontuação de Crédito",
    page_icon="💳",
    layout="wide",
)

# CSS Customizado para Design Premium
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        height: 3em;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO COM CACHE ---
@st.cache_resource
def carregar_modelo():
    try:
        # Requisito: Utilizar o arquivo 'model_final.pkl'
        with open('model_final.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# --- APLICATIVO PRINCIPAL ---
def main():
    st.title("💳 Dashboard de Análise de Risco de Crédito")
    st.write("Bem-vindo ao sistema de escoragem automática. Este dashboard utiliza um modelo de Regressão Logística avançado para prever a probabilidade de inadimplência.")
    st.markdown("---")

    # BARRA LATERAL (SIDEBAR)
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.sidebar.title("Configurações")
    
    # Requisito: Carregador de CSV
    arquivo_upload = st.sidebar.file_uploader("Carregar dados dos clientes (CSV)", type="csv")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Guia de Uso")
    st.sidebar.write("""
    1. Prepare seu arquivo CSV com os dados dos clientes.
    2. Faça o upload no campo acima.
    3. O sistema aplicará o pipeline de pré-processamento automaticamente.
    4. Analise as métricas, categorias de risco e gráficos gerados.
    5. Baixe o relatório final escorado.
    """)

    modelo = carregar_modelo()

    if modelo is None:
        st.error("🚨 Erro Crítico: O arquivo do modelo 'model_final.pkl' não foi encontrado. Certifique-se de rodar o notebook de treinamento primeiro.")
        return

    if arquivo_upload is not None:
        try:
            # Requisito: Subir um CSV
            df_entrada = pd.read_csv(arquivo_upload)
            
            # --- PRÉ-PROCESSAMENTO ---
            # Requisito: Criar um pipeline de pré-processamento (aplicado via modelo carregado)
            # Removemos colunas que não são variáveis explicativas se estiverem presentes
            colunas_remover = ['mau', 'data_ref', 'ano_mes', 'index']
            X = df_entrada.drop(columns=[c for c in colunas_remover if c in df_entrada.columns], errors='ignore')
            
            st.markdown("### 📋 Resultados da Escoragem")
            
            with st.spinner('Processando dados e calculando scores...'):
                # Requisito: Utilizar o modelo treinado para escorar a base
                probs = modelo.predict_proba(X)[:, 1]
                df_entrada['score'] = probs
                
                # Categorização de Risco
                def categorizar_risco(score):
                    if score < 0.05: return "Risco Baixo"
                    elif score < 0.10: return "Risco Medio"
                    else: return "Risco Alto"
                
                df_entrada['categoria_risco'] = df_entrada['score'].apply(categorizar_risco)

            # LINHA DE MÉTRICAS (VISUAL PREMIUM)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total de Clientes", len(df_entrada))
            m2.metric("Score Médio", f"{df_entrada['score'].mean():.2%}")
            m3.metric("Clientes de Alto Risco", len(df_entrada[df_entrada['score'] >= 0.10]))
            # Verifica se a coluna renda existe para mostrar a métrica
            renda_media = df_entrada['renda'].mean() if 'renda' in df_entrada.columns else 0
            m4.metric("Renda Média", f"R$ {renda_media:,.2f}" if renda_media > 0 else "N/A")

            # LAYOUT DE CONTEÚDO
            col_esq, col_dir = st.columns([1.5, 1])
            
            with col_esq:
                st.subheader("Tabela de Resultados (Amostra)")
                # Configuração de visualização da tabela
                cols_mostrar = ['score', 'categoria_risco'] + [c for c in df_entrada.columns if c not in ['score', 'categoria_risco']]
                st.dataframe(
                    df_entrada[cols_mostrar].head(100),
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Score de Risco",
                            help="Probabilidade de inadimplência (0 a 1)",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                    use_container_width=True
                )
            
            with col_dir:
                st.subheader("Distribuição por Categoria")
                # Gráfico de Pizza para Visualização Rápida
                fig_pizza = px.pie(
                    df_entrada, 
                    names='categoria_risco',
                    color='categoria_risco',
                    color_discrete_map={"Risco Baixo": "#28a745", "Risco Medio": "#ffc107", "Risco Alto": "#dc3545"},
                    hole=0.4,
                    template="plotly_white"
                )
                fig_pizza.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_pizza, use_container_width=True)

            # GRÁFICO DE HISTOGRAMA
            st.subheader("Histograma de Probabilidades")
            fig_hist = px.histogram(
                df_entrada, 
                x="score", 
                color="categoria_risco",
                color_discrete_map={"Risco Baixo": "#28a745", "Risco Medio": "#ffc107", "Risco Alto": "#dc3545"},
                nbins=30,
                labels={'score': 'Probabilidade de Inadimplência', 'count': 'Frequência'},
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # SEÇÃO DE EXPORTAÇÃO
            st.markdown("---")
            st.subheader("📥 Exportar Relatório Final")
            st.write("Clique no botão abaixo para baixar o arquivo CSV contendo os originais e as novas colunas de score e risco.")
            
            csv = df_entrada.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Baixar Resultados em CSV",
                data=csv,
                file_name='relatorio_risco_credito.csv',
                mime='text/csv',
                use_container_width=False
            )

        except Exception as e:
            st.error(f"❌ Erro durante o processamento: {str(e)}")
            st.info("Dica: Verifique se o seu CSV contém as mesmas colunas utilizadas no treinamento (sexo, posse_de_veiculo, renda, etc.)")

    else:
        # PÁGINA INICIAL (EMPTY STATE)
        st.info("👋 Por favor, carregue um arquivo CSV na barra lateral para iniciar a análise.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### ✨ Pré-processamento")
            st.write("Tratamento automático de nulos, outliers e dummies.")
        with c2:
            st.markdown("### 🧬 IA Integrada")
            st.write("Uso de Redução de Dimensionalidade (PCA) para maior precisão.")
        with c3:
            st.markdown("### 📊 Visualização")
            st.write("Gráficos interativos para tomada de decisão rápida.")

if __name__ == "__main__":
    main()
