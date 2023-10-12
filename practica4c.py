import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar los datos desde el archivo CSV
data = pd.read_csv("irisbin.csv")
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir un modelo de perceptrón multicapa en PyTorch
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Crear una instancia del modelo y definir la función de pérdida y el optimizador
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenar el modelo
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train))
    loss = criterion(outputs, torch.argmax(torch.FloatTensor(y_train), dim=1))
    loss.backward()
    optimizer.step()

# Evaluar el modelo en el conjunto de prueba
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test))
    predicted = torch.argmax(outputs, dim=1).numpy()
    true_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(true_labels, predicted)
    print("Accuracy on test data:", accuracy)

# Implementar los métodos leave-k-out y leave-one-out
from sklearn.model_selection import LeaveOneOut, LeavePOut
from sklearn.metrics import accuracy_score

loo = LeaveOneOut()
lpo = LeavePOut(p=2)

accuracies_loo = []
accuracies_lpo = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.argmax(torch.FloatTensor(y_train), dim=1))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test))
        predicted = torch.argmax(outputs, dim=1).numpy()
        true_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(true_labels, predicted)
        accuracies_loo.append(accuracy)

for train_index, test_index in lpo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.argmax(torch.FloatTensor(y_train), dim=1))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test))
        predicted = torch.argmax(outputs, dim=1).numpy()
        true_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(true_labels, predicted)
        accuracies_lpo.append(accuracy)

# Calcular el error esperado de clasificación, promedio y desviación estándar
error_loo = 1 - np.mean(accuracies_loo)
error_lpo = 1 - np.mean(accuracies_lpo)
average_accuracy_loo = np.mean(accuracies_loo)
average_accuracy_lpo = np.mean(accuracies_lpo)
std_accuracy_loo = np.std(accuracies_loo)
std_accuracy_lpo = np.std(accuracies_lpo)

print("Leave-One-Out Error:", error_loo)
print("Leave-P-Out (p=2) Error:", error_lpo)
print("Leave-One-Out Average Accuracy:", average_accuracy_loo)
print("Leave-P-Out (p=2) Average Accuracy:", average_accuracy_lpo)
print("Leave-One-Out Standard Deviation:", std_accuracy_loo)
print("Leave-P-Out (p=2) Standard Deviation:", std_accuracy_lpo)

# Generar una proyección en dos dimensiones de la distribución de clases
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.argmax(y, axis=1), cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Proyección en 2D de la distribución de clases (Iris Dataset)")
plt.show()
