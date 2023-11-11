import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Cargar datos del archivo CSV
auto_insurance_data = pd.read_csv("AutoInsurSweden.csv", names=["A1", "B1"])

# Reemplazar comas por puntos en la columna 'B1' y convertirla a valores numéricos
auto_insurance_data['B1'] = auto_insurance_data['B1'].str.replace(',', '.').astype(float)

# Separar las características (X) y las etiquetas (y)
X = auto_insurance_data["A1"].values.reshape(-1, 1)  # Necesita ser una matriz 2D
y = auto_insurance_data["B1"]

# Definir un umbral para la variable dependiente (y) y asignar una etiqueta binaria (0 o 1)
threshold = y.mean()
y_binary = (y > threshold).astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
linear_model = LinearRegression()

# Ajustar el modelo
linear_model.fit(X_train, y_train)

# Predecir los valores
linear_predictions = linear_model.predict(X_test)

# Redondear las predicciones a 0 o 1
linear_predictions_binary = (linear_predictions > 0.5).astype(int)



# Red Neuronal
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32)

nn_predictions = model.predict(X_test)

# Redondear las predicciones a 0 o 1
nn_predictions_binary = (nn_predictions > 0.5).astype(int)



# Evaluar el modelo de regresión lineal usando las métricas solicitadas
accuracy_linear = accuracy_score(y_test, linear_predictions_binary)
precision_linear = precision_score(y_test, linear_predictions_binary)
sensitivity_linear = recall_score(y_test, linear_predictions_binary)
specificity_linear = recall_score(y_test, linear_predictions_binary, pos_label=0)
f1_linear = f1_score(y_test, linear_predictions_binary)
cm_linear = confusion_matrix(y_test, linear_predictions_binary)

print("Accuracy (Linear Regression):", accuracy_linear)
print("Precision (Linear Regression):", precision_linear)

