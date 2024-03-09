import os
import queue
import time

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from model import Model

TRANSFORMATION_LABELS = {
    "detect": "Detect",
    "blur": "Blur",
}

BILLBOARD_MODELS_DIR = "model/billboard-detection"
billboard_models = [
    model.replace(".onnx", "") for model in os.listdir(BILLBOARD_MODELS_DIR)
]

# The general layout of the app: title -> webcam -> config panel
st.title("bart: blocking ads in real time")
webrtc_container = st.container()
configuration_panel = st.expander("Configuration", expanded=True)
stats_panel = st.expander("Estadísticas", expanded=False)

# We need a thread safe queue to store the frame processing time results
# which are processed in a different thread.
time_in_frames = queue.Queue()
# This queue is used to store the detections that we are doing en each frame
# it's used to display the area of ads seen as a % of the total area of the video.
current_frame_percentage_of_ads = queue.Queue()
total_frames = 0
rolling_average_processing_time = 0
rolling_average_percentage_of_ads = 0

# We define the model selector before we set up the rest of the app as
# we need it for the cache key
model_name = configuration_panel.selectbox(
    "Model",
    options=["Face detection", *billboard_models],
    format_func=lambda x: f"Billboard Detection: {x}" if x in billboard_models else x,
    index=0,
)

# Whenever the model name changes, we need to refresh the cached model
# We can easily see the session state with `st.write(st.session_state)`
cache_key = f"model_{model_name}"
if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = Model(
        Model.detect_billboards
        if model_name in billboard_models
        else Model.detect_faces,
        model_name=model_name,
    )
    st.session_state[cache_key] = model

# Process the frame on a different thread
def call_detect(frame):
    img = frame.to_ndarray(format="bgr24")
    start_time = time.time()
    frame, detection_area_percentage = model.detect(img, transformation=transformation, confidence=confidence / 100)
    end_time = time.time()
    time_in_frames.put(end_time - start_time)
    current_frame_percentage_of_ads.put(detection_area_percentage)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Place the webrtc_streamer on our empty container
with webrtc_container:
    webrtrc_ctx = webrtc_streamer(
        # https://github.com/whitphx/streamlit-webrtc#serving-from-remote-host
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=call_detect,
        media_stream_constraints={
            "video": {"facingMode": "environment"},
            # se puede agregar "frameRate": {"ideal": 10, "max": 15},
            "audio": False,
        },
        desired_playing_state=True,
        video_html_attrs={"controls": False, "autoPlay": True},
        key="webrtc",
    )

# Fill out the rest of the configuration panel
with configuration_panel:
    transformation = st.selectbox(
        "Transformation",
        options=("detect", "blur"),
        format_func=lambda x: TRANSFORMATION_LABELS[x],
    )
    confidence = st.slider(
        "Detection score", min_value=0, max_value=100, value=80, step=5
    )

with stats_panel:
    time_container = st.empty()
    area_percentage_container = st.empty()
    total_frames_processed_container = st.empty()

# We show the processing time per frame with a while True loop
# Everything after this block won't be run.
# Make sure this is at the end of the file.
while webrtrc_ctx.state.playing:
    total_frames += 1
    frame_processed_in = time_in_frames.get()
    last_detection_area_percentage = current_frame_percentage_of_ads.get()
    rolling_average_percentage_of_ads = (
        (rolling_average_percentage_of_ads * (total_frames - 1))
        + last_detection_area_percentage
    ) / total_frames
    time_container.text(f"Frame processing time: {frame_processed_in:.3f} seconds")
    area_percentage_container.text(f"Area covered by bounding boxes: {last_detection_area_percentage:.2f}%")
    total_frames_processed_container.text(f"Total frames processed: {total_frames}")
