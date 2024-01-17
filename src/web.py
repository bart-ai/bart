import av
import core
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("bart: blocking ads in real time")


@st.cache_resource
def cached_get_model():
    return core.get_model()


model = cached_get_model()


def call_detect(frame):
    img = frame.to_ndarray(format="bgr24")
    img = core.detect(img, model=model)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


if st.toggle("use webrtc", True):
    webrtc_streamer(
        video_frame_callback=call_detect,
        media_stream_constraints={
            "video": {"facingMode": "environment"},
            # se puede agregar "frameRate": {"ideal": 10, "max": 15},
            "audio": False,
        },
        # desired_playing_state=True,  # arrancar en playing, pero perdes el uso del otro input
        video_html_attrs={"controls": False, "autoPlay": True},
        key="webrtc",
    )
else:
    FRAME_WINDOW = st.image([])
    usebackcamera = False
    backcamera = cv2.VideoCapture(1)
    if backcamera.read()[0]:
        usebackcamera = True
    backcamera.release()

    camera = cv2.VideoCapture(1) if usebackcamera else cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Stop webrtc before opening the camera.")

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = core.detect(frame, model=model)
        FRAME_WINDOW.image(frame)
