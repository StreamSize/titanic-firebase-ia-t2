"""Entrenamiento reproducible: árbol de decisión para Titanic (Kaggle)."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
CSV = BASE / 'titanic-train.csv'
MODEL_FILE = BASE / 'modelo_titanic.joblib'
METRICS_FILE = BASE / 'resultados_modelo.json'
NUMERICAS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
CATEGORICAS = ['Sex', 'Embarked']
VARIABLES = NUMERICAS + CATEGORICAS


def entrenar():
    df = pd.read_csv(CSV)
    obligatorias = set(VARIABLES + ['Survived'])
    faltantes = obligatorias - set(df.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas en el CSV: {sorted(faltantes)}')
    df = df.dropna(subset=['Survived']).copy()
    X = df[VARIABLES]
    y = df['Survived'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocesamiento = ColumnTransformer([
        ('numericas', SimpleImputer(strategy='median'), NUMERICAS),
        ('categoricas', Pipeline([
            ('imputar', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAS),
    ])
    pipeline = Pipeline([
        ('preprocesamiento', preprocesamiento),
        ('arbol', DecisionTreeClassifier(random_state=42)),
    ])
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid={
            'arbol__max_depth': [2, 3, 4, 5, 6, 8, None],
            'arbol__min_samples_leaf': [1, 3, 5, 10],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=1,
        refit=True,
    )
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_
    pred = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1])
    profundidad = modelo.named_steps['arbol'].get_depth()
    metricas = {
        'dataset': 'Titanic Kaggle - titanic-train.csv',
        'total_registros': int(len(df)),
        'registros_entrenamiento': int(len(X_train)),
        'registros_prueba': int(len(X_test)),
        'profundidad_maxima_seleccionada': busqueda.best_params_['arbol__max_depth'],
        'profundidad_real': int(profundidad),
        'min_samples_leaf': int(busqueda.best_params_['arbol__min_samples_leaf']),
        'exactitud_validacion_cruzada': round(float(busqueda.best_score_), 4),
        'exactitud_prueba': round(float(accuracy_score(y_test, pred)), 4),
        'f1_prueba': round(float(f1_score(y_test, pred)), 4),
        'matriz_confusion_orden_no_si': matriz.tolist(),
        'reporte_clasificacion': classification_report(y_test, pred, output_dict=True),
    }
    # Solo cargar archivos joblib creados y controlados por este proyecto.
    joblib.dump({'modelo': modelo, 'variables': VARIABLES, 'metricas': metricas}, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=== ENTRENAMIENTO TITANIC COMPLETADO ===')
    print('Registros:',len(df))
    print('Mejor max_depth:', metricas['profundidad_maxima_seleccionada'])
    print('Exactitud prueba:', f"{metricas['exactitud_prueba']:.2%}")
    print('Matriz [0=No, 1=Sí]:', matriz.tolist())
    return metricas


if __name__ == '__main__':
    entrenar()
