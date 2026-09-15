import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

X = np.array([
    [1,1,4.8,120,5,400],
    [1,0,4.2,30,2,100],
    [0,1,4.9,300,10,900],
    [1,1,3.0,5,0,10],
    [0,0,4.5,200,7,600],
    [1,1,5.0,80,6,300],
    [1,0,2.5,10,1,20],
    [0,1,4.7,150,4,500]
])

y = np.array([1,0,1,0,0,1,0,1])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, verbose=0)

class Cliente:
  def __init__(self, nome, cidade, categoria_desejada):
      self.nome = nome
      self.cidade = cidade
      self.categoria_desejada = categoria_desejada

class Profissional:
  def __init__(self, nome, cidade, categoria, aval_media, num_avaliacoes, experiencia_anos, idade_conta_dias):
     self.nome = nome
     self.cidade = cidade
     self.categoria = categoria
     self.aval_media = aval_media
     self.num_avaliacoes = num_avaliacoes
     self.experiencia_anos = experiencia_anos
     self.idade_conta_dias = idade_conta_dias

cliente1 = Cliente("Ana", "SP", "eletricista")

prof1 = Profissional("Carlos", "SP", "eletricista", 4.8, 120, 6, 400)
prof2 = Profissional("João", "RJ", "encanador", 4.2, 40, 3, 200)
prof3 = Profissional("Maria", "SP", "eletricista", 4.9, 250, 10, 900)

lista_profissionais = [prof1, prof2, prof3]

def match_ia(cliente, profissional, model, scaler):

    mesma_cidade = 1 if cliente.cidade == profissional.cidade else 0
    mesma_categoria = 1 if cliente.categoria_desejada == profissional.categoria else 0

    entrada = np.array([[
        mesma_cidade,
        mesma_categoria,
        profissional.aval_media,
        profissional.num_avaliacoes,
        profissional.experiencia_anos,
        profissional.idade_conta_dias
    ]])

    entrada = scaler.transform(entrada)
    prob = model.predict(entrada, verbose=0)

    return prob[0][0]

def recomendar_profissionais(cliente, lista_profissionais, model, scaler):

    ranking = []

    for prof in lista_profissionais:
        score = match_ia(cliente, prof, model, scaler)
        ranking.append((prof.nome, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

ranking = recomendar_profissionais(cliente1, lista_profissionais, model, scaler)

for nome, score in ranking:
    print(nome, "-> chance de match:", round(score, 3))
