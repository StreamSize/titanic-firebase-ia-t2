# Titanic IA + Firebase Realtime Database (actividad 2)

Este proyecto cumple la actividad de clasificación de Titanic con un **árbol de decisión**, **aplicación web**, **servicio externo Render** y **Firebase Realtime Database**.

## Qué contiene
- `titanic-train.csv`: conjunto Titanic de Kaggle (891 pasajeros) con columna objetivo `Survived`.
- `entrenar_modelo.py`: preprocesamiento, entrenamiento, validación cruzada, elección de profundidad y evaluación con prueba separada.
- `app.py`: interfaz web de Gradio para ingresar clase, edad, género, SibSp, Parch, Fare y Embarked; predice y guarda en Firebase.
- `modelo_titanic.joblib`: modelo entrenado previamente para prueba local. **En Render se vuelve a entrenar en build** para asegurar compatibilidad entre versiones.
- `resultados_modelo.json`: evaluación calculada de este dataset y este modelo.
- `.python-version`: usa Python 3.12 en Render.
- `render.yaml`: opcional para automatizar configuración del servicio.
- `GUIA_DESDE_CERO.md` y `GUION_VIDEO.md`: pasos de configuración y guion del video.

## Entrenamiento verificado en esta versión
- Registros: 891 (entrenamiento 712; prueba 179).
- Árbol de decisión: mejor `max_depth=4` por validación cruzada estratificada en entrenamiento, con `min_samples_leaf` seleccionado.
- Exactitud en prueba: 79,33 % (142/179).
- Matriz de confusión con orden [no, sí]: `[[104, 6], [31, 38]]`.
- Precisión estimada para pasajeros nuevos, no un hecho histórico.

## Inicio rápido local

Con Python 3.12 instalado, abrir Terminal (PowerShell) dentro de esta carpeta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe entrenar_modelo.py
.\.venv\Scripts\python.exe app.py
```

Abrir http://127.0.0.1:7860 en navegador. Sin configurar Firebase, la app calcula la predicción pero **indica que NO se guardó**.

Si el equipo solo tiene Python 3.14, recomendamos instalar Python 3.12 para evitar posibles problemas con dependencias.

## Publicar en Render

GitHub → repositorio nuevo (sin archivos secretos) → subir archivos descomprimidos → Render → New → Web Service → conectar repositorio → seleccionar **Free**.

- Runtime / Language: Python 3
- Build Command: `pip install -r requirements.txt && python entrenar_modelo.py`
- Start Command: `python app.py`
- Archivo `.python-version`: `3.12`

Render usa `$PORT`; `app.py` ya enlaza `0.0.0.0:$PORT`.

En **Environment** agregar secretos:

- `FIREBASE_DATABASE_URL` = URL exacta de Realtime Database, p. ej. `https://proyecto-default-rtdb.firebaseio.com` (solo como ejemplo, NO copiar literalmente).
- `FIREBASE_SERVICE_ACCOUNT_JSON` = contenido completo del JSON de cuenta de servicio descargado desde Firebase. Mantenerlo secreto, jamás agregarlo al repositorio.

El código hace `db.reference('predicciones_titanic').push(registro)` y devuelve el identificador del registro **solo después de que Firebase confirme el guardado**.

## Seguridad
- El JSON de cuenta de servicio es **privado**, nunca ponerlo en código, GitHub o capturas del video.
- Realtime Database se puede iniciar en **modo bloqueado**; el Admin SDK autenticado del servidor puede escribir sin exponer las reglas públicas.
- Para presentación conviene retirar/deshabilitar el despliegue público después de grabar el video.
- No añadas datos personales reales de terceros al formulario; el dataset es histórico.

## Fuentes de datos / documentación
- Dataset: https://www.kaggle.com/competitions/titanic/data
- Esquema CSV público: https://github.com/agconti/kaggle-titanic/blob/master/data/train.csv
- Firebase Realtime Database: https://firebase.google.com/docs/database/admin/start
- Render: https://render.com/docs/web-services

**Importante:** según el enunciado, al profesor se le entrega **solo un video**; el ZIP es material de preparación y código, no el formato final de entrega.
