# bart: blocking ads in real time

<!-- TODO: SCREENSHOT DESKTOP Y MOBILE-->
<!-- TODO: MAIN DESCRIPTION AND OBJECTIVE AND AIM -->

This project consists of two components: training models for billboard detection, and deploying them live for use on the web at https://bart.fly.dev/

The website ([src](./src/)) serves as a simple way of testing the different models trained on the [yolo8](./yolo8/) directory. It lets you test the different models inside the [src/model](./src/model) dir and allows you to select the image transformation to apply on the detections (such as blurring the video wherever a billboard appears). It also provides some profiling tools to test the accuracy and performance of each model.

The model training directory ([yolo8](./yolo8)) contains several tools to train the experiments with different parameters and get the final models for the website to use.
