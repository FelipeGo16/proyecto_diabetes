from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid", context="notebook")

app = Flask(__name__)

# ================================
#   CARGAR Y ENTRENAR EL MODELO
# ================================
df = pd.read_csv("diabetes.csv", header=None)
df.columns = ['preg','plas','pres','skin','test','mass','pedi','age','Class']

df = df.drop(df.index[0])      # Eliminar encabezado duplicado
df = df.drop_duplicates()       # Quitar duplicados

#Entrenamiento de random forest con scikit-learn
# Separamos en X e Y
X = df.drop('Class', axis=1)
y = df['Class']

#entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Definir los hiperparámetros y sus posibles valores
param_grid = {
    'n_estimators': [10, 25, 50],
    'max_depth': [5, 10, 15],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Instancia del modelo
model = RandomForestClassifier(random_state=42)

# Crear el objeto GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Ajustar el modelo con GridSearchCV
grid.fit(X_train, y_train)

# Obtener el modelo con el mejor rendimiento
best_model = grid.best_estimator_

# Mejores parametros del modelo
grid.best_params_

# ================================
#   IMPRESIONES POR CONSOLA
# ================================
print("\n==============================")
print(" PRIMERAS FILAS DEL DATAFRAME")
print("==============================")
print(df.head())

print("\n==============================")
print(" PARAMETROS DEL MODELO")
print("==============================")
print(grid.best_params_)

print("\n==============================")
print(" SHAPES DE ENTRENAMIENTO Y TEST")
print("==============================")
print("X_train:", X_train.shape, " | y_train:", y_train.shape)
print("X_test :", X_test.shape, " | y_test :", y_test.shape)

# Predicciones internas para métricas
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print("\n==============================")
print(" ACCURACIES DEL MODELO")
print("==============================")
print(f"Accuracy TRAIN: {accuracy_train*100:.2f}%")
print(f"Accuracy TEST : {accuracy_test*100:.2f}%")

# Obtener los scores del GridSearch
k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=42)

# Nota: best_model -> es el modelo ya optimizado por GridSearchCV
scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')

print("\n==============================")
print(" VALIDACIÓN CRUZADA")
print("==============================")
print("Scores de cada pliegue:", scores * 100)
print("Promedio accuracy:", scores.mean() * 100)
print("Desviación estándar:", scores.std() * 100)
print("==============================\n")


# ================================
#   FUNCIONES
# ================================

def plot_importancia_variables():
    fig, ax = plt.subplots(figsize=(10, 5))
    importancias = best_model.feature_importances_
    
    sns.barplot(x=X.columns, y=importancias, ax=ax)
    ax.set_title("Importancia de las variables en la predicción")

    for i, val in enumerate(importancias):
        ax.text(i, val, f"{val:.2f}", ha='center', va='bottom')

    img = io.BytesIO()
    plt.tight_layout()
    fig.savefig(img, format='png')
    img.seek(0)
    grafica = base64.b64encode(img.getvalue()).decode()

    plt.close(fig)
    return grafica

grafica_importancia = plot_importancia_variables()

# ================================
#   FUNCIÓN: PEDIGREE (DPF)
# ================================

def diabetes_pedigree_function(familiares):
    """
    Calcula la Diabetes Pedigree Function (DPF)
    familiares: lista de tuplas (generacion, tiene_diabetes)
       - generacion: nivel de parentesco (1=padres, 2=abuelos/tíos, 3=primos, etc.)
       - tiene_diabetes: 1 si el familiar tiene diabetes, 0 si no
    """
    if not familiares:
        return 0     # Si no se ingresan familiares, el riesgo se asume 0

    pesos = [1 / (2 ** g) for g, _ in familiares]
    contribuciones = [peso * d for peso, (_, d) in zip(pesos, familiares)]

    dpf = sum(contribuciones) / sum(pesos)
    return round(dpf, 3)

# ================================
#            RUTAS
# ================================

@app.route("/")
def formulario():
    return render_template("index.html")

# --------------------------------------
# NUEVA RUTA PARA CALCULAR PEDI (DPF)
# --------------------------------------
@app.route("/predecir", methods=["POST"])
def predecir():
    # Obtener valores excepto PETI (que ahora será calculado)
    columnas_sin_pedi = ['preg','plas','pres','skin','test','mass','age']
    valores = [float(request.form[col]) for col in columnas_sin_pedi]

    # =========================
    # Cálculo del DPF (PEDI)
    # =========================
    generacion = request.form.getlist("generacion[]")
    diabetes = request.form.getlist("diabetes[]")

    familiares = [(int(g), int(d)) for g, d in zip(generacion, diabetes)]
    pedi_calculado = diabetes_pedigree_function(familiares)

    # Insertar el valor PEDI en la posición correcta
    valores.insert(6, pedi_calculado)

    # Crear dataframe del usuario
    df_usuario = pd.DataFrame([valores], columns=X.columns)

    # Predicción del modelo
    prediccion = best_model.predict(df_usuario)[0]

    resultado = "Probablemente tiene diabetes" if prediccion == '1' else "Probablemente NO tiene diabetes"

    return render_template("resultados.html",
                           resultado=resultado,
                           pedi=pedi_calculado,
                           grafica=grafica_importancia)



# ================================
#           MAIN
# ================================
if __name__ == "__main__":
    app.run(debug=True)
