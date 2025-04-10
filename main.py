import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados
try:
    data = pd.read_excel("df_completo.xlsx")
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

# Remover registros com preco = 0
data = data[data['preco'] > 0]

# Separar variáveis independentes e dependentes
X = data.drop(columns=['preco'])
y = data['preco']

# Normalizar 'area_construcao'
scaler = StandardScaler()
if 'area_construcao' in X.columns:
    X['area_construcao'] = scaler.fit_transform(X[['area_construcao']])

# Selecionar o modelo
model_option = st.selectbox("Escolha o modelo de regressão:", ["Regressão Linear", "Random Forest", "Gradient Boosting"])

if model_option == "Regressão Linear":
    model = LinearRegression()
elif model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Treinar o modelo
model.fit(X, y)

# Avaliar o modelo
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Criar interface no Streamlit
st.title("Previsão de Preço de Casas")
st.write("Insira os atributos da casa para prever o preço")

# Exibir métricas do modelo
st.subheader("Métricas do Modelo")
st.write(f"Erro Médio Absoluto (MAE): {mae:,.2f}")
st.write(f"Erro Quadrático Médio (MSE): {mse:,.2f}")
st.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:,.2f}")
st.write(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Criar entradas para o usuário
area_construcao = st.number_input("Área Construída (m²)", min_value=10, max_value=1000, value=100)
quartos = st.number_input("Número de Quartos", min_value=1, max_value=10, value=2)
banheiros = st.number_input("Número de Banheiros", min_value=1, max_value=10, value=1)
garagem_coberta = st.number_input("Vagas na Garagem Coberta", min_value=0, max_value=5, value=1)

# Criar select box para o código do bairro
bairros_unicos = sorted(data['bairro_num'].unique())
bairro_num = st.selectbox("Código do Bairro", bairros_unicos)

tipo_casa_check = st.checkbox("É uma Casa?")
tipo_casa = 1 if tipo_casa_check else 0
tipo_predio = 1 if not tipo_casa_check else 0

# Entrada adicional para previsao_classificacao_qualidade
qualidade = st.selectbox("Classificação de Qualidade", [1, 2, 3])

# Normalizar o input da área construída
data_input = pd.DataFrame([[area_construcao, quartos, banheiros, garagem_coberta, bairro_num, tipo_casa, tipo_predio, qualidade]],
                          columns=['area_construcao', 'quartos', 'banheiros', 'garagem_coberta', 'bairro_num', 'tipo_casa', 'tipo_predio', 'previsao_classificacao_qualidade'])

data_input['area_construcao'] = scaler.transform(data_input[['area_construcao']])

# Verificar o número de colunas de entrada
st.write("Número de colunas do X:", X.shape[1])
st.write("Número de colunas do input:", data_input.shape[1])

# Botão para previsão
if st.button("Prever Preço"):
    try:
        preco_previsto = model.predict(data_input)[0]
        st.success(f"O preço estimado da casa é R$ {preco_previsto:,.2f}")
    except Exception as e:
        st.error(f"Erro na previsão: {e}")