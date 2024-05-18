# bart: web

The bart website is powered by https://streamlit.io/ and deployed onto https://fly.io/

The billboard models are all in the [`ONNX` format](https://onnx.ai/) and can be read with either [OpenCV](https://docs.opencv.org) or [onnxruntime](https://onnxruntime.ai/) (which is set as the default runtime, as it is the most performant of the two of them). The website also provides a minimal [face detection](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) model to test some features more easily, and the Ultralytics pre-trained models on the [Open Images v7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) dataset to use as a benchmark to compare to.

The main component, [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc), is used to easily read the video stream from the user's webcam and process it through our inference models. WebRTC allows for real-time video processing and requires a server to relay the traffic between the user client and our streamlit server. Our deployed server is provided by [Open Relay](https://www.metered.ca/tools/openrelay/).

## Run locally

```sh
# on a venv
$ pip install -r requirements.txt
$ streamlit run web.py
```

## Deployment

On each push to `master` the CI automatically deploys the new version by running the project [Dockerfile](../Dockerfile) and reading the specs from the project [fly.toml](../fly.toml).

The fly.io machine already has the `BART_METERED_LIVE_SECRET_KEY` environment variable set up so that the deployed version can use the correct STUN+TURN servers. Without the proper credentials, it defaults to a free stun server.

To manually deploy, install the [fly command-line tool](https://fly.io/docs/hands-on/install-flyctl/) and run `flyctl deploy` after authentication.
