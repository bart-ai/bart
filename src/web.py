import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

import core

TRANSFORMATION_LABELS = {
    "detect": "Detect",
    "blur": "Blur",
}

st.title("bart: blocking ads in real time")


@st.cache_resource
def cached_get_model():
    return core.Model(core.Model.detect_billboards)


model = cached_get_model()


def call_detect(frame):
    img = frame.to_ndarray(format="bgr24")
    img = model.detect(img, transformation=transformation, confidence=confidence / 100)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
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
    transformation = st.selectbox(
        "Transformation",
        options=("detect", "blur"),
        format_func=lambda x: TRANSFORMATION_LABELS[x],
    )
    confidence = st.slider(
        "Detection score", min_value=0, max_value=100, value=80, step=5
    )