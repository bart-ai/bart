import os
import queue
import time

import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from model import TRANSFORMATIONS, Model

BILLBOARD_MODELS_DIR = "model/billboard-detection"
billboard_models = [
    model.replace(".onnx", "") for model in os.listdir(BILLBOARD_MODELS_DIR)
]

# The general layout of the app: title -> webcam -> config panel
st.title("bart: blocking ads in real time")
webrtc_container = st.container()
configuration_panel = st.expander("Configuration", expanded=True)
stats_panel = st.container(border=True)

# We need a thread safe queue to store the frame metadata results
# which are processed in a different thread.
stats_queue = queue.Queue()
total_frames = 0

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
# As errors don't bubble up from this thread, it's prudent to to keep track of
#   them by printing them
def call_detect(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        start_time = time.time()
        frame, detection_area_percentage = model.detect(img, transformation=transformation, confidence=confidence / 100)
        end_time = time.time()
        frame_processed_in = end_time - start_time
        stats_queue.put([frame_processed_in, detection_area_percentage])
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print(f"Error on detection thread: {e}")


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
        options=TRANSFORMATIONS,
        format_func=lambda x: x.capitalize(),
    )
    confidence = st.slider(
        "Detection score", min_value=0, max_value=100, value=80, step=5
    )

with stats_panel:
    if st.toggle('Enable profiling'):
        timetab, areatab = st.tabs(['Performance Profiling', 'Detections Profiling'])
        with timetab:
            st.text("Frame processing time")
            time_line_chart = st.line_chart(pd.DataFrame(columns=['time']))
            time_stats = st.empty()

        with areatab:
            st.text("Percentage of area covered by bounding boxes")
            area_line_chart = st.line_chart(pd.DataFrame(columns=['percentage']))
            area_stats = st.empty()

        while webrtrc_ctx.state.playing:
            total_frames += 1
            [frame_processed_in, last_detection_area_percentage] = stats_queue.get()

            time_line_chart.add_rows(pd.DataFrame([{"time": frame_processed_in}]))
            area_line_chart.add_rows(pd.DataFrame([{"percentage": last_detection_area_percentage}]))

            time_stats.code(f"""
                            Number of Frames Processed: {total_frames}
                            Current Frame Processing Time: {frame_processed_in:.3f} seconds
            """)

            area_stats.code(f"""
                            Number of Frames Processed: {total_frames}
                            Current Area Covered by Bounding Boxes: {last_detection_area_percentage:.3f} percent
            """)
