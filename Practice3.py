import numpy as np
import math  # Importa el módulo math para la función math.exp

def calcular_gradiente(funcion, x, epsilon=1e-5):
    """
    Calcula el gradiente de una función en un punto dado utilizando diferenciación numérica.

    :param funcion: La función que se desea derivar.
    :param x: El punto en el que se calculará el gradiente.
    :param epsilon: El tamaño del paso para la aproximación de la derivada.
    :return: El gradiente de la función en el punto dado.
    """
    gradient = np.zeros_like(x)
    
    for i in range(len(x)):
        x_i_plus_epsilon = x.copy()
        x_i_minus_epsilon = x.copy()
        
        x_i_plus_epsilon[i] += epsilon
        x_i_minus_epsilon[i] -= epsilon
        
        gradient[i] = (funcion(x_i_plus_epsilon) - funcion(x_i_minus_epsilon)) / (2 * epsilon)
    
    return gradient

# Definición de la función de ejemplo (f(x_1, x_2) = 10 - e^(-(x_1^2 + 3*x_2^2)))
def funcion_ejemplo(x):
    return 10 - math.exp(-(x[0]**2 + 3*x[1]**2))

# Punto en el que se calculará el gradiente
x_punto = np.array([1.0, 2.0])

# Definir un punto inicial y otros hiperparámetros
punto_inicial = np.array([1.0, 2.0])
tasa_aprendizaje = 0.1
num_iteraciones = 100

# Inicializar el punto actual para el descenso del gradiente
x_actual = punto_inicial

# Realizar el descenso del gradiente para encontrar el mínimo global
for _ in range(num_iteraciones):
    gradiente_resultante = calcular_gradiente(funcion_ejemplo, x_actual)
    x_actual -= tasa_aprendizaje * gradiente_resultante

# Punto mínimo global encontrado
punto_minimo_global = x_actual

print("Gradiente en el punto {}: {}".format(x_punto, gradiente_resultante))
print("Punto mínimo global encontrado:", punto_minimo_global)
print("Valor mínimo global encontrado:", funcion_ejemplo(punto_minimo_global))
