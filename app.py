from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import os
import joblib

# 1. Carregar os dados clássicos do Iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. Inicializar e treinar o modelo de Machine Learning
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# 3. Salvar o modelo treinado em um arquivo binário
joblib.dump(modelo, "modelo_iris.joblib")

print("Modelo treinado e salvo com sucesso como 'modelo_iris.joblib'!")

st.set_page_config(
    page_title="Projeto Machine Learning - Iris",
    page_icon="🌸",
    layout="wide"
)


# Configuração inicial da página
st.set_page_config(
    page_title="Classificador de Flores Iris",
    page_icon="🌸",
    layout="centered"
)

# ==========================================
# CARREGAMENTO DO MODELO (Com Cache do Streamlit)
# ==========================================
@st.cache_resource
def carregar_modelo():
    # Verifica se o arquivo do modelo existe antes de carregar
    if os.path.exists("modelo_iris.joblib"):
        return joblib.load("modelo_iris.joblib")
    else:
        return None

modelo = carregar_modelo()

# Título principal
st.title("🌸 Classificador de Flores Iris")

st.write(
    """
    Este aplicativo recebe as características de uma flor Iris e
    utiliza um modelo de Machine Learning treinado para adivinhar
    a espécie da planta.
    """
)

st.divider()

# Seção explicativa
st.header("Informe as características da flor")

st.info(
    """
    Preencha os valores abaixo com as medidas da flor em centímetros.
    
    O modelo Iris utiliza quatro características:
    - Comprimento da sépala
    - Largura da sépala
    - Comprimento da pétala
    - Largura da pétala
    """
)

# Criando duas colunas para deixar o layout mais bonito
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Comprimento da sépala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Comprimento da pétala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Largura da sépala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Largura da pétala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

st.divider()

# Mostrando os dados informados pelo usuário
st.subheader("Dados informados")

dados_iris = {
    "Comprimento da sépala": sepal_length,
    "Largura da sépala": sepal_width,
    "Comprimento da pétala": petal_length,
    "Largura da pétala": petal_width
}

st.write(dados_iris)

# Mapeamento numérico das classes do dataset Iris original
especies_dict = {
    0: "Iris-setosa 🌸",
    1: "Iris-versicolor 🌿",
    2: "Iris-virginica 🌺"
}

# Botão para fazer a previsão
botao_adivinhar = st.button("🔍 Adivinhar espécie da flor", type="primary")

if botao_adivinhar:
    if modelo is None:
        st.error(
            "O arquivo do modelo (`modelo_iris.joblib`) não foi encontrado! "
            "Execute o script de treinamento primeiro."
        )
    else:
        # Organizando os dados no formato esperado pelo modelo (matriz 2D)
        entrada_modelo = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Realizando a previsão com o modelo carregado pelo Joblib
        previsao_id = modelo.predict(entrada_modelo)[0]
        especie_prevista = especies_dict.get(previsao_id, "Desconhecida")

        # Exibindo o resultado da predição
        st.success(f"Espécie prevista pelo modelo: **{especie_prevista}**")

        # Opcional: exibir as probabilidades de cada classe
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada_modelo)[0]
            st.write("Confiança do modelo por classe:")
            for idx, prob in enumerate(probabilidades):
                nome_esp = especies_dict.get(idx, f"Classe {idx}")
                st.progress(float(prob), text=f"{nome_esp}: {prob * 100:.2f}%")

st.divider()

# Rodapé didático
st.caption(
    "Projeto didático com Streamlit + Scikit-Learn + Joblib — Dataset Iris"
)