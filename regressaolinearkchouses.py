# -*- coding: utf-8 -*-
"""Passo 1: Importar dados do CSV com o Pandas.
"""

import pandas as pd
dataFrame = pd.read_csv("/content/kc_house_data.csv")
print("Quantidade de linhas:", len(dataFrame))
print("Quantidade de colunas", len(dataFrame.columns))

"""Passo 2: vamos fazer uma análise basica das 5 primeiras linhas do data frame."""

dataFrame.head()

"""Passo 2.1: vamos pedir para o Pandas exibir dados sobre o dataset."""

dataFrame.info()

"""Passo 2.2: vamos pedir para o Pandas exibir detalhes estatístico relevantes associados ao conjunto de dados."""

dataFrame.describe()

"""Passo 3: ao aplicar Regressão Linear é interessante fazer uma análise de correlação de variáveis. Primeiro, vamos fazer uma análise gráfica."""

import seaborn as sb
sb.pairplot(dataFrame,
            x_vars=["bedrooms","bathrooms","sqft_living"],
            y_vars=["price"])

"""Passo 4: agora que conhecemos os dados, vamos fazer a Regressão Linear. No entanto, começaremos com uma regressão linear simples, onde iremos prever o preço do imóvel com base no tamanho. Como vimos antes, o primeiro é separar as variáveis dependente e independente."""

#Variável INDEPENDENTE
X=dataFrame[["sqft_living"]]
X

#variável DEPENDENTE
y = dataFrame[["price"]]
y

"""Passo 5: estudamos anteriormente que, para evitar o overfitting, uma boa prática é separar dados para treinamento e dados para teste."""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10)
print("Quantidade de registros separados para treinamento:",len(X_train))
print("Quantidade de registros separados para teste:", len(X_test))

"""Passo 6: agora que separamos os dados para treinamento, vamos treinar o modelo. O primeiro passo é importar o algoritmo de Regressão Linear e criar o objeto que fará a Regressão."""

from sklearn.linear_model import LinearRegression
objRegressaoLinear = LinearRegression()

"""Passo 7: agora que temos um objeto apto a aplicar Regressão Linear, vamos treinar o modelo."""

objRegressaoLinear.fit(X_train,y_train)

"""Passo 8: vamos pedir para a regressão prever o preço de casas com tamanhos que iremos informar."""

data = [
          [1181],
          [2575],
          [775],
          [450]
        ]
objRegressaoLinear.predict(data)

"""Passo 9: Pergunta que não que calar: cadê o gráfico gerado? Vamos pedir para o Python exibir."""

y_pred = objRegressaoLinear.predict(X_train)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred, color="red")

"""Passo 10: Para verificar a qualidade da regressão linear, aconselha-se medir duas métricas: o MSE e o R2. Devemos buscar regressões onde o MSE seja o menor possível e o R2 seja maior do que 0.7."""

y_pred_test = objRegressaoLinear.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test,y_pred_test))
print("R2:",r2_score(y_test,y_pred_test))

"""Passo 11: aprendemos como gerar uma Regressão Linear simples, que utiliza uma variável independente apenas para prever o valor da dependente.
Agora, vamos tentar melhorar a regressão, para isso, utrilizaremos três variáveis independentes para prever o preço. No caso, vamos utilizar as seguintes variáveis independentes: sqft_living,bedrooms, bathrooms.
"""

X2 = dataFrame[["sqft_living","bedrooms","bathrooms"]]
y2 = dataFrame[["price"]]

"""Passo 12: como sabemos, é importante separar dados de teste dos de treinamento."""

from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.10)
print("Quantidade de registos separados para treinamento:", len(X2_train))
print("Quantidade de registros separados para teste:", len(X2_test))

"""Passo 13: Agora que os dados de treinamento e teste foram separados, vamos treinar o modelo."""

from sklearn.linear_model import LinearRegression
objRegressao2 = LinearRegression()
objRegressao2.fit(X2_train,y2_train)

"""Passo 14: uma vez que a Regressão Linear foi gerada, utilizando-se três variáveis INDEPENDENTES para prever a DEPENDENTE, vamos medir a acurácia do modelo."""

y2_pred_test = objRegressao2.predict(X2_test)

from sklearn.metrics import r2_score, mean_squared_error
print("R2:", r2_score(y2_test,y2_pred_test))
print("MSE:", mean_squared_error(y2_test,y2_pred_test))

"""Passo final: Recaptulando, primeiro geramos uma regressão utilizando UMA variável INDEPENDENTE para prever o preço. Despois, geramos uma regressão que utilizou TRÊS variáveis INDEPENDENTES para prever o preço. Agora, vamos gerar um modelo com o máximo de variáveis INDEPENDENTES possível."""

#variáveis INDEPENTES
X3 = dataFrame.drop(columns=["id","date","price"])
X3

#variável DEPENDENTES
y3 = dataFrame[["price"]]
y3

"""Como vimos, é conveniente separarmos dados de teste e de treinamento."""

from sklearn.model_selection import train_test_split
X3_train,X3_test,y3_train,y3_test = train_test_split(X3,y3,test_size=0.10)
print("Quantidade de registros separados para treinamento:", len(X3_train))
print("Quantidade de registros separados para teste:",len(X3_test))

"""Agora vamos treinar o modelo."""

from sklearn.linear_model import LinearRegression
objRegressao3 = LinearRegression()
objRegressao3.fit(X3_train,y3_train)

"""Agora é hora de calcular o R2 e o MSE."""

y3_pred_test = objRegressao3.predict(X3_test)
from sklearn.metrics import r2_score, mean_squared_error
print("R2:", r2_score(y3_test,y3_pred_test))
print("MSE:",mean_squared_error(y3_test,y3_pred_test))

"""Para finalizar, vamos exibir um gráfico que compara o preço real das casas separadas para teste com o preço previsto pelo modelo."""

y3_test["price_predict"] = y3_pred_test
dfAnalise = y3_test.head(15)
dfAnalise.plot(kind="bar")