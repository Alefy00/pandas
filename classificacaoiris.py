
"""Vamos iniciar o desenvolvimento de uma arvore de decisão para classificação de Iris Para isso, vamos carregar  os dados csv no python utilizando a biblioteca pandas"""

import pandas as pd
nomeColunas =["alturaSepala", "larguraSepalas", "alturaPetala", "larguraPetala", "classe"]
dataFrame = pd.read_csv("/content/iris.data", names=nomeColunas)
print("Quantidade de linhas do data frame: " , len(dataFrame))
print("Quantidade de colunas do data Frame: " , len(dataFrame.columns))

"""Agora que os dados foram importados para o dataframe vamos ver os primeiros registros da tabela"""

dataFrame.head(150)

"""Vamos verificar as ultimas linhas da tabela no data frame"""

dataFrame.tail()

"""detalhes da estrutura usando Pandas"""

dataFrame.info()

"""vamos pedir para o pandas exibir uma brave analise estatistica dos dados do dataframe como: medida, desvio padrão, valores maximo e valores minimo."""

dataFrame.describe()

"""as vezes, os atributos que temos na base não são satisfatorios para contornar isso, podemos criar features geradas a partir da combinação de atributos  existentes"""

dataFrame["areaSepala"]=dataFrame["alturaSepala"]*dataFrame["larguraSepalas"]
dataFrame["areaPetala"]=dataFrame["alturaPetala"]*dataFrame["larguraPetala"]

dataFrame.head()

X = dataFrame[["alturaSepala", "larguraSepala", "areaSepala", "alturaPetala", "larguraPetala","areaPelata"]]
y = dataFrame["classe"]