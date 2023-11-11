import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ucimlrepo import fetch_ucirepo

# Fetch dataset
zoo = fetch_ucirepo(id=111)

# Extract features and target
X = zoo.data.features
y = zoo.data.targets['type'].to_numpy()  # Convertir a array NumPy

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train, y_train)
y_pred_lr = logistic_regression_model.predict(X_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

# Support Vector Machines
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Neural Network
model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
_, accuracy_nn = model.evaluate(X_test, y_test)
y_pred_nn = (model.predict(X_test) > 0.5).astype(int).reshape(-1)

# Evaluate each model
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted', zero_division=1))
    print("Recall (Sensitivity):", recall_score(y_true, y_pred, average='weighted'))

    # Use confusion_matrix for multiclass classification
    cm = confusion_matrix(y_true, y_pred)

    # Calculate specificity and F1 Score differently for multiclass
    if len(np.unique(y_true)) == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:  # Multiclass classification
        specificity = 0  # Not defined for multiclass
    print("Specificity:", specificity)

    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))

# Evaluate each model
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)
evaluate_model("Support Vector Machines", y_test, y_pred_svm)
evaluate_model("Naive Bayes", y_test, y_pred_nb)
evaluate_model("Neural Network", y_test, y_pred_nn)