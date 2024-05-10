# -*- coding: utf-8 -*-
"""Passo 1: Importar os dados do csv para um data frame"""

import pandas as pd
dataFrame = pd.read_csv ("/content/Mall_Customers.csv")
print("Quantidade de linhas:", len(dataFrame))
print("Quantidade de colunas", len(dataFrame.columns))

"""Passo 2: Vamos fazer uma analise exploratoria da base, vamos visualizando os primeiros registros"""

dataFrame.head()

"""Passo 3: vamos verificar informações gerais"""

dataFrame.info()

"""Passo 4: vamos solicitar as estatisticas da base"""

dataFrame.describe()

"""Verificar se tem algum valor nulo na base"""

dataFrame.isnull().sum()

"""Vamos verificar quantos clientes tem de cada sexo"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(dataFrame['Age'], bins=6)

dataFrame.corr()

"""Vimos que a coluna gender abrange valores textuais. para aplicar IA é coerente focar na utilizaçao de dados em formato numerico."""

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataFrame ['Gender'] = label_encoder.fit_transform(dataFrame['Gender'])

dataFrame.head()

"""agora que o dataset esta bem formatado vamos separar os dados que serao ultilizados"""

X=dataFrame[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X