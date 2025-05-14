# K-Vecinos más Cercanos (K-NN)

#Explicación de los cambios:
#Entrenamiento y prueba: El script ahora está mejor documentado para que los estudiantes entiendan cómo dividir los datos en conjuntos de entrenamiento y prueba, y cómo realizar la predicción.
#Escalado de características: Se ha añadido un paso para normalizar los datos con StandardScaler, lo cual es importante para mejorar el rendimiento de los algoritmos basados en distancias, como el K-NN.
#Matriz de confusión: Se genera la matriz de confusión y se calcula la precisión del modelo para evaluar su rendimiento.
#Visualización: La visualización del conjunto de entrenamiento y prueba ahora incluye una visualización del modelo ajustado sobre la malla de características, lo que ayuda a entender cómo el modelo hace las predicciones.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargando el dataset
dataset = pd.read_csv('data.csv')  # Lee el archivo CSV con los datos
X = dataset[['moisture', 'temp']].values  # Características
y = dataset['pump'].values  # Objetivo (0: OFF, 1: ON)

# Dividiendo el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print("Conjunto de entrenamiento (X_train):")
print(X_train)
print("Etiquetas de entrenamiento (y_train):")
print(y_train)
print("Conjunto de prueba (X_test):")
print(X_test)
print("Etiquetas de prueba (y_test):")
print(y_test)

# Escalado de características (Feature Scaling)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Ajuste y transformación para el conjunto de entrenamiento
X_test = sc.transform(X_test)  # Transformación para el conjunto de prueba
print("Conjunto de entrenamiento escalado (X_train):")
print(X_train)
print("Conjunto de prueba escalado (X_test):")
print(X_test)

# Entrenando el modelo K-NN sobre el conjunto de entrenamiento
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)  # Ajusta el modelo a los datos de entrenamiento

# Predicción de un nuevo caso: humedad = 55, temperatura = 30
resultado = classifier.predict(sc.transform([[55, 30]]))
print(f"Predicción para humedad 55 y temperatura 30: {'ON' if resultado[0] == 1 else 'OFF'}")

# Predicción sobre el conjunto de prueba
y_pred = classifier.predict(X_test)  # Predice las etiquetas para el conjunto de prueba
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))  # Compara las predicciones con las etiquetas reales

# Creando la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)  # Genera la matriz de confusión
print("Matriz de Confusión:")
print(cm)
print("Precisión del modelo:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # Muestra la precisión del modelo

# Visualización - Conjunto de entrenamiento
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 5, stop=X_set[:, 0].max() + 5, step=0.5),
    np.arange(start=X_set[:, 1].min() - 5, stop=X_set[:, 1].max() + 5, step=0.5)
)
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')), edgecolor='k')
plt.title('K-NN (Conjunto de Entrenamiento)')
plt.xlabel('Humedad del Suelo')
plt.ylabel('Temperatura')
plt.legend(['Pump OFF', 'Pump ON'])
plt.show()

# Visualización - Conjunto de prueba
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 5, stop=X_set[:, 0].max() + 5, step=0.5),
    np.arange(start=X_set[:, 1].min() - 5, stop=X_set[:, 1].max() + 5, step=0.5)
)
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')), edgecolor='k')
plt.title('K-NN (Conjunto de Prueba)')
plt.xlabel('Humedad del Suelo')
plt.ylabel('Temperatura')
plt.legend(['Pump OFF', 'Pump ON'])
plt.show()
