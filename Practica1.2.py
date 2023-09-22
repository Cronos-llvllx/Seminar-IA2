import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# Lee el archivo CSV
df = pd.read_csv("spheres1d10.csv", names=['A1', 'B1', 'C1', 'D1'])


# Solicitar la cantidad de particiones
num_partitions = int(input("Ingrese la cantidad de particiones: "))

# Solicitar el porcentaje de patrones de entrenamiento y generalización
p_train = float(input("Ingrese el porcentaje de patrones de entrenamiento (0-1): "))
p_generalization = float(input("Ingrese el porcentaje de patrones de generalización (0-1): "))

# Ciclo para crear las particiones
for i in range(num_partitions):
    # Divide en conjunto de entrenamiento y generalización
    train, generalization = train_test_split(df, train_size=p_train, test_size=p_generalization, random_state=i)

    # Selecciona las columnas 'A1', 'B1' y 'C1' como características (X)
    X_train = train[['A1', 'B1', 'C1']]
    X_generalization = generalization[['A1', 'B1', 'C1']]
    
    # Selecciona la columna 'D1' como variable objetivo (y)
    y_train = train['D1']
    y_generalization = generalization['D1']
    
    # Entrena el perceptrón simple
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    
    # Evalúa el perceptrón en el conjunto de generalización
    accuracy = perceptron.score(X_generalization, y_generalization)
    
    print(f"Partición {i + 1}:")
    print("Ejemplos usados para entrenar:", len(train))
    print("Ejemplos usados para generalización:", len(generalization))
    print(f"Precisión en generalización: {accuracy:.2f}")
    print("=" * 40)
