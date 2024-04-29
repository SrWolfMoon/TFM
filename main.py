import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import warnings

def cargar_datos(nombre_archivo):
    return pd.read_csv(nombre_archivo)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Cargar datos de entrenamiento
    datos_entrenamiento = cargar_datos('datos_entrenamiento.csv')

    # Eliminar columnas 'Nombre' y 'Apellido'
    datos_entrenamiento = datos_entrenamiento.drop(['Nombre', 'Apellido', 'Id'], axis=1)

    # Convertir columnas de lesiones a valores booleanos
    for columna in ['LesionRodilla', 'LesionTobillo', 'LesionHombro']:
        datos_entrenamiento[columna] = datos_entrenamiento[columna].map({'Si': True, 'No': False})

    # Dividir datos en características (X) y etiquetas (y)
    X = datos_entrenamiento.drop(['LesionRodilla', 'LesionTobillo', 'LesionHombro'], axis=1)
    y = (datos_entrenamiento['LesionRodilla'] | datos_entrenamiento['LesionTobillo'] | datos_entrenamiento['LesionHombro'])

    # Combinar características de entrenamiento y prueba para asegurar la misma codificación one-hot
    X_combined = pd.get_dummies(X, columns=['Genero', 'EntrenoPistaBuena'])

    # Dividir datos combinados en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


    # Construir la pipeline con selección de características y validación cruzada
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_combined.columns.tolist())  # Adjust num_features according to your dataset
    ])

    # Define pipeline with preprocessor and RandomForestClassifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Define parameters for GridSearchCV
    parametros = {
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }


    # Realizar la búsqueda en cuadrícula con validación cruzada
    grid_search = GridSearchCV(pipeline, parametros, cv=5, scoring='accuracy')
    
    print("Preparando el modelo con el conjunto de test")
    # Entrenar el modelo
    grid_search.fit(X_train, y_train)

    print("Evaluando el modelo")
    # Evaluar el modelo
    accuracy = grid_search.score(X_test, y_test)
    print("Accuracy del modelo:", accuracy*100, "%")

    # Leer datos para predicción desde otro CSV
    datos_prediccion = cargar_datos('datos_prediccion.csv')
    datos_prediccion = datos_prediccion.drop(['Id','Nombre', 'Apellido'], axis=1)
    datos_prediccion = pd.get_dummies(datos_prediccion, columns=['Genero', 'EntrenoPistaBuena'])
    
    columnas_faltantes = set(['Genero_Femenino', 'Genero_Masculino', 'EntrenoPistaBuena_No', 'EntrenoPistaBuena_Si']) - set(datos_prediccion.columns)
    for columna in columnas_faltantes:
        datos_prediccion[columna] = False

    print("DATOS A PREDECIR: ")
    print(datos_prediccion.head(5))
    
    # Predecir las posibilidades de lesión
    posibilidades_lesion = grid_search.predict_proba(datos_prediccion)
    for i, probabilidad in enumerate(posibilidades_lesion):
        probabilidad_redondeada = round(probabilidad[1] * 100, 2)
        print(f"Probabilidad de lesión de la persona {i+1}: {probabilidad_redondeada}%")