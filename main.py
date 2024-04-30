import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def cargar_datos(filename):
    """Carga datos desde un archivo CSV especificado."""
    return pd.read_csv(filename)

def preparar_datos(datos):
    """Prepara los datos eliminando columnas innecesarias y mapeando valores."""
    datos = datos.drop(['Nombre', 'Apellido', "Id"], axis=1)
    for columna in ['LesionRodilla', 'LesionTobillo', 'LesionHombro']:
        datos[columna] = datos[columna].map({'Si': True, 'No': False})
    return datos

def dividir_datos(datos):
    """Divide los datos en características y etiquetas."""
    X = datos.drop(['LesionRodilla', 'LesionTobillo', 'LesionHombro'], axis=1)
    y = datos['LesionRodilla'] | datos['LesionTobillo'] | datos['LesionHombro']
    X = pd.get_dummies(X, columns=['Genero', 'EntrenoPistaBuena'])
    return X, y

def main():
    warnings.filterwarnings("ignore")
    
    # Cargar y preparar datos de entrenamiento
    datos_entrenamiento = cargar_datos('datos_entrenamiento.csv')
    datos_entrenamiento = preparar_datos(datos_entrenamiento)
    X, y = dividir_datos(datos_entrenamiento)

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=48)
    
    # Crear pipeline de procesamiento y clasificación
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X.columns.tolist())])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=48))
    ])

    # Parámetros para búsqueda en cuadrícula
    parametros = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Configurar y entrenar GridSearchCV
    grid_search = GridSearchCV(pipeline, parametros, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Evaluar el modelo
    accuracy = grid_search.score(X_test, y_test)
    print(f"Accuracy del modelo: {accuracy * 100:.2f}%")

    # Leer y preparar datos de predicción
    datos_prediccion = cargar_datos('datos_prediccion.csv')
    datos_prediccion = datos_prediccion.drop(['Id', 'Nombre', 'Apellido'], axis=1)
    datos_prediccion = pd.get_dummies(datos_prediccion, columns=['Genero', 'EntrenoPistaBuena'])

    # Asegurar que todas las columnas necesarias estén presentes
    columnas_faltantes = set(['Genero_Femenino', 'Genero_Masculino', 'EntrenoPistaBuena_No', 'EntrenoPistaBuena_Si']) - set(datos_prediccion.columns)
    for columna in columnas_faltantes:
        datos_prediccion[columna] = 0

    print("DATOS A PREDECIR: ")
    print(datos_prediccion.head(5))
    
    # Predecir probabilidades de lesiones
    posibilidades_lesion = grid_search.predict_proba(datos_prediccion)
    for i, probabilidad in enumerate(posibilidades_lesion):
        print(f"Probabilidad de lesión de la persona {i + 1}: {probabilidad[1] * 100:.2f}%")

if __name__ == "__main__":
    main()
