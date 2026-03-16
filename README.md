# Projeto Final: Dashboard de Credit Scoring 💳

Este repositório contém o projeto final do módulo de **Credit Scoring**, integrando Ciência de Dados e Desenvolvimento Web com Streamlit.

## 📺 Demonstração em Vídeo

O vídeo abaixo demonstra as principais funcionalidades do Dashboard:

[**Assista à Demonstração do Projeto (Clique aqui para abrir o vídeo)**](./demo_video.mp4)

---

## 🚀 Funcionalidades Principal
- **Escoragem em Tempo Real**: Carregamento de arquivos CSV para cálculo instantâneo de risco.
- **Pipeline de Pré-processamento**: Tratamento automático de nulos, outliers e dummies.
- **Visualização de Risco**: Gráficos interativos (Pizza e Histograma) com categorias de risco calibradas.
- **Exportação de Dados**: Geração de relatórios CSV escorados.

## 📂 Estrutura do Repositório
- **`app.py`**: Aplicativo Streamlit da ferramenta de escoragem.
- **`Mod38Projeto.ipynb`**: Notebook principal com EDA, Modelagem e Avaliação.
- **`Mod38Exercicio1 (1).ipynb`**: Exercício prático do módulo.
- **`model_final.pkl`**: Pipeline final do scikit-learn serializado.
- **`test_data.csv`**: Base de teste para validação imediata.
- **Exercícios Anteriores**: Notebooks dos módulos 34 a 37 também incluídos para composição do portfólio.

## 🛠️ Como Executar

### Pré-requisitos
- Python 3.12+
- Dependências listadas no `requirements.txt` (ou instaladas via pip):
  ```bash
  pip install streamlit pandas scikit-learn numpy plotly joblib pyarrow imbalanced-learn lightgbm
  ```

### Execução
No terminal, dentro da pasta do projeto:
```bash
streamlit run app.py
```

---
**Desenvolvido como projeto final de módulo.**
