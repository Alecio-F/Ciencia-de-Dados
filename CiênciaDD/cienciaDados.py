import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

tabela = pd.read_csv('advertising.csv')

# Criando gráfico
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)

# Exibindo gráfico
# plt.show()

y = tabela['Vendas']
x = tabela[['TV', 'Radio', 'Jornal']]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
print(f'Previsão do modelo da árvore de decisão {r2_score(y_teste, previsao_arvoredecisao):.2f}%')
print(f'Previsão do modelo da regressão linear {r2_score(y_teste, previsao_regressaolinear):.2f}%')

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsão Regressiva Linear'] = previsao_regressaolinear
tabela_auxiliar['Previsão Árvore de Decisão'] = previsao_arvoredecisao

plt.figure(figsize=(15, 5))
sns.lineplot(data=tabela_auxiliar)
# plt.show()

tabela_n = pd.read_csv('novos.csv')
previsao = modelo_arvoredecisao.predict(tabela_n)
print()
print(f'Previsão de vendas usando a tabela de novos.csv {previsao}')

plt.show()