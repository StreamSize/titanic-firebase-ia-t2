"""Predicción de supervivencia del Titanic con Firebase Realtime Database."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
MODELO = BASE / 'modelo_titanic.joblib'
if not MODELO.exists():
    # Primera ejecución: entrena con el CSV incluido.
    from entrenar_modelo import entrenar
    entrenar()
PAQUETE = joblib.load(MODELO)  # Archivo local generado por nuestro entrenamiento
CLASIFICADOR = PAQUETE['modelo']
METRICAS = PAQUETE['metricas']


def guardar_firebase(registro):
    """Devuelve (id, mensaje). Nunca declara éxito si push() falla."""
    llave_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON', '').strip()
    url = os.environ.get('FIREBASE_DATABASE_URL', '').strip()
    if not llave_json or not url:
        return None, ('Firebase NO conectado: configura FIREBASE_SERVICE_ACCOUNT_JSON y '
                      'FIREBASE_DATABASE_URL en Render. La predicción se calculó, '
                      'pero NO se guardó en Realtime Database.')
    try:
        import firebase_admin
        from firebase_admin import credentials, db
        try:
            firebase_admin.get_app()
        except ValueError:
            cuenta = json.loads(llave_json)
            cred = credentials.Certificate(cuenta)
            firebase_admin.initialize_app(cred, {'databaseURL': url})
        ref = db.reference('predicciones_titanic').push(registro)
        if not ref.key:
            raise RuntimeError('Firebase no devolvió una clave de registro')
        return ref.key, f'✅ Guardado confirmado en Firebase Realtime Database. ID: {ref.key}'
    except Exception as exc:
        # No mostrar claves privadas ni configuraciones sensibles en pantalla.
        print('Error al guardar en Realtime Database:', type(exc).__name__)
        return None, ('❌ No se pudo guardar en Firebase. Revisa URL, credenciales '
                      'y permisos en los registros del servicio.')


def predecir_y_guardar(clase, edad, genero, hermanos, padres, tarifa, puerto):
    if clase is None or edad is None or genero is None or hermanos is None or padres is None or tarifa is None or puerto is None:
        return '⚠️ Completa todos los campos.', '', 'No se generó ningún registro.'
    if not 0 <= float(edad) <= 110 or not 0 <= float(tarifa) <= 1000:
        return '⚠️ Edad o tarifa fuera del rango permitido.', '', 'No se guardó ningún registro.'
    if not 0 <= int(hermanos) <= 20 or not 0 <= int(padres) <= 20:
        return '⚠️ Cantidad de familiares no válida.', '', 'No se guardó ningún registro.'
    datos = {
        'Pclass': int(clase),
        'Age': float(edad),
        'Sex': str(genero),
        'SibSp': int(hermanos),
        'Parch': int(padres),
        'Fare': float(tarifa),
        'Embarked': str(puerto),
    }
    df = pd.DataFrame([datos], columns=PAQUETE['variables'])
    prediccion = int(CLASIFICADOR.predict(df)[0])
    probabilidad = float(CLASIFICADOR.predict_proba(df)[0][1])
    resultado = '✅ Probablemente SOBREVIVIÓ' if prediccion else '❌ Probablemente NO SOBREVIVIÓ'
    registro = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'modelo': 'Arbol de decision',
        'datos_pasajero': datos,
        'sobrevivio_predicho': prediccion,
        'resultado': 'Sobrevivió' if prediccion else 'No sobrevivió',
        'probabilidad_supervivencia': round(probabilidad, 4),
    }
    uid, estado = guardar_firebase(registro)
    detalle = f'Probabilidad estimada de sobrevivir: {probabilidad:.1%}.'
    if uid:
        detalle += f' Registro guardado: predicciones_titanic/{uid}'
    return resultado, detalle, estado


with gr.Blocks(title='Titanic IA | Supervivencia y Firebase') as demo:
    gr.Markdown('# 🚢 Titanic IA: ¿Sobrevivió el pasajero?')
    gr.Markdown('Modelo de **árbol de decisión** entrenado con el conjunto de Titanic de Kaggle. '
                'Al pulsar el botón, el resultado se intenta guardar en **Firebase Realtime Database**.')
    with gr.Row():
        with gr.Column():
            clase = gr.Dropdown([('Primera clase',1),('Segunda clase',2),('Tercera clase',3)],
                               value=3, label='Clase del pasaje')
            edad = gr.Number(value=22, label='Edad (años)', minimum=0, maximum=110)
            genero = gr.Dropdown([('Masculino','male'),('Femenino','female')],
                                 value='male', label='Género')
            hermanos = gr.Number(value=0, precision=0, minimum=0, maximum=20,
                                 label='Hermanos/cónyuges a bordo (SibSp)')
        with gr.Column():
            padres = gr.Number(value=0, precision=0, minimum=0, maximum=20,
                               label='Padres/hijos a bordo (Parch)')
            tarifa = gr.Number(value=7.25, minimum=0, maximum=1000,
                               label='Tarifa del boleto (Fare)')
            puerto = gr.Dropdown([('Southampton','S'),('Cherbourg','C'),('Queenstown','Q')],
                                 value='S', label='Puerto de embarque')
            boton = gr.Button('🔎 Predecir y guardar en Firebase', variant='primary')
    resultado = gr.Textbox(label='Predicción', interactive=False)
    detalle = gr.Textbox(label='Probabilidad e ID del registro', interactive=False)
    estado = gr.Textbox(label='Estado del guardado en Realtime Database', interactive=False, lines=2)
    gr.Markdown(f'**Evaluación del modelo:** exactitud en prueba {METRICAS["exactitud_prueba"]:.1%}; '
                f'profundidad máxima elegida {METRICAS["profundidad_maxima_seleccionada"]}. '
                'Predicción estadística para una actividad educativa, no un hecho histórico.')
    boton.click(predecir_y_guardar, inputs=[clase, edad, genero, hermanos, padres, tarifa, puerto],
                outputs=[resultado, detalle, estado])

if __name__ == '__main__':
    puerto_servidor = int(os.environ.get('PORT', '7860'))
    demo.launch(server_name='0.0.0.0', server_port=puerto_servidor, show_error=False)
