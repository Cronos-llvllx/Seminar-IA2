import numpy as np
import matplotlib.pyplot as plt

# Función para leer patrones desde un archivo CSV
def leer_patrones(archivo):
    data = np.genfromtxt(archivo, delimiter=',')
    entradas = data[:, :-1]
    salidas = data[:, -1]
    return entradas, salidas

# Función para entrenar un perceptrón
def entrenar_perceptron(entradas, salidas, tasa_aprendizaje, max_epocas):
    num_entradas = entradas.shape[1]
    pesos = np.random.rand(num_entradas)  # Inicialización aleatoria de pesos
    errores = []

    for epoca in range(max_epocas):
        error_epoca = 0
        for i in range(entradas.shape[0]):
            entrada = entradas[i]
            salida_deseada = salidas[i]
            salida_calculada = np.dot(entrada, pesos)
            error = salida_deseada - salida_calculada
            pesos += tasa_aprendizaje * error * entrada
            error_epoca += abs(error)
        errores.append(error_epoca)

        # Criterio de finalización: detenerse si el error es cero
        if error_epoca == 0:
            break

    return pesos, errores

# Función para probar el perceptrón entrenado
def probar_perceptron(entradas, pesos):
    salidas = []
    for entrada in entradas:
        salida_calculada = np.dot(entrada, pesos)
        salidas.append(salida_calculada)
    return salidas

# Función para graficar los patrones y la recta de separación
def graficar_patrones_y_recta(entradas, salidas, pesos):
    plt.scatter(entradas[:, 0], entradas[:, 1], c=salidas)
    x = np.linspace(-2, 2, 100)
    y = (-pesos[0] * x) / pesos[1]
    plt.plot(x, y, '-r', label='Recta de Separación')
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.legend(loc='upper left')
    plt.show()

# Lectura de patrones de entrenamiento
entradas_entrenamiento, salidas_entrenamiento = leer_patrones('XOR_trn.csv')

# Parámetros de entrenamiento
tasa_aprendizaje = 0.1
max_epocas = 10000

# Entrenamiento del perceptrón
pesos_entrenados, errores_entrenamiento = entrenar_perceptron(entradas_entrenamiento, salidas_entrenamiento, tasa_aprendizaje, max_epocas)

# Prueba del perceptrón entrenado en datos reales
entradas_prueba, salidas_prueba = leer_patrones('XOR_tst.csv')
salidas_calculadas = probar_perceptron(entradas_prueba, pesos_entrenados)

# Graficar patrones y recta de separación
graficar_patrones_y_recta(entradas_prueba, salidas_calculadas, pesos_entrenados)
