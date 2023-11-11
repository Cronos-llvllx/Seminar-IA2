import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Cargar datos del archivo CSV usando punto y coma como delimitador y omitir la primera fila
data_wine_quality = pd.read_csv("winequality-white.csv", delimiter=";")

# Asegúrate de que la columna esté correctamente formateada con puntos en lugar de comas

# Separar las características (X) y las etiquetas (y)
X = data_wine_quality.iloc[-1, 1].values  # Necesita ser una matriz 2D
y = data_wine_quality["A1"].values
print(data_wine_quality.columns)


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

accuracy_logistic = accuracy_score(y_test, logistic_predictions)
precision_logistic = precision_score(y_test, logistic_predictions, average='micro')
recall_logistic = recall_score(y_test, logistic_predictions, average='micro')
f1_logistic = f1_score(y_test, logistic_predictions, average='micro')
cm_logistic = confusion_matrix(y_test, logistic_predictions)

print("Logistic Regression Accuracy:", accuracy_logistic)


# 2. K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, knn_predictions)
precision_knn = precision_score(y_test, knn_predictions, average='micro')
recall_knn = recall_score(y_test, knn_predictions, average='micro')
f1_knn = f1_score(y_test, knn_predictions, average='micro')
cm_knn = confusion_matrix(y_test, knn_predictions)

print("K-Nearest Neighbors Accuracy:", accuracy_knn)



# 3. Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
naive_bayes_predictions = naive_bayes_model.predict(X_test)

accuracy_naive_bayes = accuracy_score(y_test, naive_bayes_predictions)
precision_naive_bayes = precision_score(y_test, naive_bayes_predictions, average='micro')
recall_naive_bayes = recall_score(y_test, naive_bayes_predictions, average='micro')
f1_naive_bayes = f1_score(y_test, naive_bayes_predictions, average='micro')
cm_naive_bayes = confusion_matrix(y_test, naive_bayes_predictions)

print("Naive Bayes Accuracy:", accuracy_naive_bayes)



# 4. Red Neuronal (usando TensorFlow/Keras)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
model.evaluate(X_test, y_test)

nn_predictions = model.predict(X_test).round()


print("Neural Network Accuracy:", accuracy_nn)
