# Cómo entrenar el modelo

Para entrenar un modelo, hay que correr el archivo `train.py` con el modelo deseado.

El modelo es una versión de YOLO8, podemos cambiar la versión a entrenar desde el objeto de modelos.

Una vez elegido el modelo, se puede correr el archivo y esperar. Si se usa una `API_KEY` de comet
nos va pasar la URL dónde podemos ver el progreso.

Para cambiar el training set hay que cambiar `YOLO_DATA_YML_PATH` para que apunte al training set que querramos.

# Cómo predecir una imagen

Para probar el modelo (rapidamente) solo hay que correr el archivo `predict.py`.

Dentro de este archivo está el path a la imagen a predecir.
