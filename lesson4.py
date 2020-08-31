# Задание 1
# Импортируйте библиотеки pandas, numpy и matplotlib.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# Масштабируйте данные с помощью StandardScaler.
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# Постройте диаграмму рассеяния на этих данных.
# 

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

boston = load_boston()

feature_names = boston.feature_names
target = boston.target

X = pd.DataFrame(boston.data, columns=feature_names)
y = pd.DataFrame(target, columns=['price'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
X_test_tsne = tsne.fit_transform(X_test_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.show()

#
# Задание 2
# С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# используйте все признаки из датафрейма X_train.
# Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# и раскрасьте точки из разных кластеров разными цветами.
# Вычислите средние значения price и CRIM в разных кластерах.
#
from sklearn.cluster import KMeans

model_Kmeans = KMeans(n_clusters=3, random_state=42, max_iter=100)
labels = model_Kmeans.fit_predict(X_train_tsne)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels)
plt.scatter(model_Kmeans.cluster_centers_[:, 0], model_Kmeans.cluster_centers_[:, 1], marker='D', edgecolors='red')
plt.show()

X_all = pd.DataFrame(X_train)
X_all = X_all.merge(y_train, left_index=True, right_index=True)
X_all = X_all.merge(pd.DataFrame(labels, columns=['label_cluster']), left_index=True, right_index=True)

data_mean_bar = pd.DataFrame(columns=['mPrice', 'mCRIM'])

for i in numpy.unique(labels):
    mean_price = X_all[X_all['label_cluster'] == i][['price']].mean().price
    mean_CRIM = X_all[X_all['label_cluster'] == i][['CRIM']].mean().CRIM
    tmp =  pd.DataFrame([[mean_price,mean_CRIM]], columns=['mPrice', 'mCRIM'])
    data_mean_bar = data_mean_bar.append(other=tmp, ignore_index=True)
    print(f"Label_{i} MEAN_PRICE: {mean_price} MEAN_CRIM: {mean_CRIM}")


plt.style.use('fivethirtyeight')
data_mean_bar.plot(kind='bar')
plt.yscale(value='log')
plt.show()

# * Задание 3
# Примените модель KMeans, построенную в предыдущем задании,
# к данным из тестового набора.
# Вычислите средние значения price и CRIM в разных кластерах на тестовых данных.
