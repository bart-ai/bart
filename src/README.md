# bart-ai: web

The bart-ai web is powered by https://streamlit.io/ and deployed onto https://fly.io/.

## Run locally

```sh
# on a venv
$ pip install -r requirements.txt
$ streamlit run web.py
```

## Deployment

On each push to `master` the CI automatically deploys the new version to https://bart.fly.dev/, by running the project [Dockerfile](../Dockerfile) and reading the specs from the project [fly.toml](../fly.toml).

To manually deploy, install the [fly command-line tool](https://fly.io/docs/hands-on/install-flyctl/) and run `fly deploy` after authentication.

## Script

Additionally, [script.py](./script.py) is a helpful command-line tool to test some OpenCV functionalities and the core idea of the project, separate from the web.

```sh
$ python3 script.py ../test-videos/face.mp4
Summary: processed 19.67 seconds of video in 21.89 seconds (13.47 fps)
    Input ../test-videos/face.mp4: 590 frames  @ 30.0 fps
    Output ..a/test-videos/out-face.mp4: 295 frames  @ 15.0 fps
```

```sh
$ python3 script.py --help
usage: script.py [-h] [--target-fps TARGET_FPS] [--secs-refetch SECS_REFETCH] [--confidence CONFIDENCE] video

Video object detector

positional arguments:
  video                 video file to process

options:
  -h, --help            show this help message and exit
  --target-fps TARGET_FPS
                        Frames per second of the output video
                        Lower values result in better performance
                        (default: 15)
  --secs-refetch SECS_REFETCH
                        Number of seconds before refetching the model
                        Objects detected by the model are tracked between refetches
                        (example: 0 == there is no tracking involved)
                        (example: 2 == the model is refetched every 2 seconds)
                        (default: 0)
  --confidence CONFIDENCE
                        Confidence threshold to use when detecting objects
                        (example: 0.15 == 15% confidence)
                        (default: 0.8)
```
