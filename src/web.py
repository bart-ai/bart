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

st.title("bart: blocking ads in real time")

time_in_frames = queue.Queue()


def call_detect(frame):
    img = frame.to_ndarray(format="bgr24")
    start_time = time.time()
    model = Model(Model.detect_faces)
    img = model.detect(img, transformation=transformation, confidence=confidence / 100)
    end_time = time.time()
    time_in_frames.put(end_time - start_time)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


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

with st.expander("Configuration"):
    time_container = st.empty()
    transformation = st.selectbox(
        "Transformation",
        options=("detect", "blur"),
        format_func=lambda x: TRANSFORMATION_LABELS[x],
    )
    confidence = st.slider(
        "Detection score", min_value=0, max_value=100, value=80, step=5
    )


# Everything after this block won't be run.
# Make sure this is at the end of the file.
while webrtrc_ctx.state.playing:
    result = time_in_frames.get()
    time_container.text(f"Frame processing time: {result:.3f} seconds")
