# Cómo entrenar el modelo

Para entrenar un modelo, hay que correr el archivo `train.py` con el modelo deseado.

El modelo es una versión de YOLO8, podemos cambiar la versión a entrenar desde el objeto de modelos.

Una vez elegido el modelo, se puede correr el archivo y esperar. Si se usa una `API_KEY` de comet
nos va pasar la URL dónde podemos ver el progreso.

Para cambiar el training set hay que cambiar `YOLO_DATA_YML_PATH` para que apunte al training set que querramos.

# Cómo predecir una imagen

Para probar el modelo (rapidamente) solo hay que correr el archivo `predict.py`.

Dentro de este archivo está el path a la imagen a predecir.

# Cómo exportar el modelo para usar en la web-app

Una vez que tenemos el modelo entrenado y queremos utilizarlo en la web-app, podemos hacer uso del script `convert_to_onnx.py`.

Esto nos va a generar un modelo de tipo `.onnx` que podemos mover dentro de `src/model` para poder utilizarlo desde la web.
