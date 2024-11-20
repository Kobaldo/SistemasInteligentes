import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Dados ampliados
dados = np.array([
    [150, 0], [120, 1], [200, 2], [100, 1], [250, 2],
    [140, 0], [110, 1], [210, 2], [130, 0], [240, 2]
])
rotulos = np.array([0, 1, 2, 1, 2, 0, 1, 2, 0, 2])

# Normalização
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(dados)

# Modelo revisado
modelo = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(6, activation='relu'),
    Dense(3, activation='softmax')
])

# Configuração e treinamento
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelo.fit(dados_normalizados, rotulos, epochs=50)

# Teste com normalização
teste = np.array([[180, 2]])  # Dado de teste
teste_normalizado = scaler.transform(teste)
previsao = modelo.predict(teste_normalizado)

# Interpretação dos resultados
classes = ['Maçã', 'Banana', 'Laranja']  # Nome das classes
probabilidades = previsao[0]  # Probabilidades previstas para cada classe

# Exibição clara dos resultados
print("Probabilidades de cada classe:")
for i, prob in enumerate(probabilidades):
    print(f"- {classes[i]}: {prob:.2%}")  # Exibe em porcentagem com 2 casas decimais

classe_predita = np.argmax(previsao)  # Índice da classe com maior probabilidade
print(f"\nClasse mais provável: {classes[classe_predita]} com {probabilidades[classe_predita]:.2%} de chance.")
