import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Cargar los datos y seleccionar las columnas relevantes
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Definir las características (X) y el objetivo (y)
predict = "G3"

X = data.drop(columns=[predict]).to_numpy()
print(f'X is: {type(X)}')

y = data[predict].to_numpy()
print(f'y is: {type(y)}')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

print(f'X_train is: {type(X_train)}')
print(f'X_test is: {type(X_test)}')     
print(f'y_train is: {type(y_train)}')
print(f'y_test is: {type(y_test)}')

# Crear y entrenar el modelo usando un pipeline
# El pipeline primero escala los datos y luego aplica la regresión lineal
linear = make_pipeline(StandardScaler(with_mean=False), linear_model.LinearRegression())
linear.fit(X_train, y_train)

# Evaluar la precisión del modelo con los datos de prueba
precision = linear.score(X_test, y_test)
print(f"Precisión del modelo (R^2): {precision:.2%}")

# Obtener los coeficientes y el intercepto del modelo de regresión lineal dentro del pipeline
coefficients = linear.steps[1][1].coef_
intercept = linear.steps[1][1].intercept_

print('Coeficientes: \n', coefficients)
print('Intercepto: \n', intercept)

def predecir_g3(g1, g2, studytime, failures, absences):
    """
    Predice la nota final (G3) de un alumno basándose en sus datos.

    Args:
        g1 (int): Nota del primer periodo.
        g2 (int): Nota del segundo periodo.
        studytime (int): Tiempo de estudio semanal (1-4).
        failures (int): Número de suspensos anteriores.
        absences (int): Número de faltas.

    Returns:
        float: La predicción de la nota G3.
    """
    # El pipeline se encarga de escalar los nuevos datos automáticamente
    nuevo_alumno = np.array([[g1, g2, studytime, failures, absences]])
    prediccion_g3 = linear.predict(nuevo_alumno)
    return prediccion_g3[0]

# Ejemplo de uso de la función de predicción
g1_nuevo = 10
g2_nuevo = 10
studytime_nuevo = 1
failures_nuevo = 1
absences_nuevo = 5

g3_predicha = predecir_g3(g1_nuevo, g2_nuevo, studytime_nuevo, failures_nuevo, absences_nuevo)

print(f"\n--- Predicción para un alumno nuevo ---")
print(f"Datos del alumno: G1={g1_nuevo}, G2={g2_nuevo}, StudyTime={studytime_nuevo}, Failures={failures_nuevo}, Absences={absences_nuevo}")
print(f"La nota final G3 predicha es: {g3_predicha:.2f}")
