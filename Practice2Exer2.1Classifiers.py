import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Cargar datos del archivo CSV
data_diabetes = pd.read_csv("pima-indians-diabetes.csv", names=["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1"])

# Análisis exploratorio de datos (EDA)
print(data_diabetes.describe())
print(data_diabetes.info())

# Dividir los datos en características (X) y etiquetas (y)
X = data_diabetes.drop("I1", axis=1)
y = data_diabetes["I1"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 2. Support Vector Machines (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, svm_predictions)
precision_svm = precision_score(y_test, svm_predictions)
recall_svm = recall_score(y_test, svm_predictions)
f1_svm = f1_score(y_test, svm_predictions)
cm_svm = confusion_matrix(y_test, svm_predictions)

print("Support Vector Machines Accuracy:", accuracy_svm)
print("Precision (SVM):", precision_svm)
print("Recall (SVM):", recall_svm)
print("F1 Score (SVM):", f1_svm)
print("Confusion Matrix (SVM):\n", cm_svm)


# 3. Red Neuronal (usando TensorFlow/Keras)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16)

nn_predictions = model.predict(X_test).round()

accuracy_nn = accuracy_score(y_test, nn_predictions)
precision_nn = precision_score(y_test, nn_predictions)
recall_nn = recall_score(y_test, nn_predictions)
f1_nn = f1_score(y_test, nn_predictions)
cm_nn = confusion_matrix(y_test, nn_predictions)

print("Red Neuronal Accuracy:", accuracy_nn)
print("Precision (Neural Network):", precision_nn)
print("Recall (Neural Network):", recall_nn)
print("F1 Score (Neural Network):", f1_nn)
print("Confusion Matrix (Neural Network):\n", cm_nn)

# Estos son los resultados de SVM y la Red Neuronal usando las métricas.
