import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import queue
import time

# 1. Page Configuration
st.set_page_config(page_title="SLR System - UoM", layout="wide")

# Official CSS for Side-by-Side Layout and Top-Controls
st.markdown("""
    <style>
    /* Force WebRTC controls (Start/Stop) to the TOP of the video */
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        display: flex;
        flex-direction: column-reverse;
    }
    
    /* Camera Styling (Small and Professional) */
    .stVideo {
        max-width: 260px !important;
        border: 2px solid #1E3A8A;
        border-radius: 8px;
        margin-top: 5px;
    }
    
    /* Header and Department centering */
    .header-text {
        text-align: center;
        margin-bottom: 20px;
    }

    /* Fixed Footer for the output */
    .fixed-footer {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background-color: #0F172A; 
        color: #10B981;
        text-align: center; 
        padding: 10px;
        font-size: 22px; 
        font-weight: bold;
        z-index: 1000; 
        border-top: 2px solid #1E3A8A;
    }

    /* Button and Slider alignment */
    .stButton>button {
        width: 100%;
        background-color: #DC2626;
        color: white;
        border-radius: 4px;
        height: 45px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Model & Backend Setup
@st.cache_resource
def load_yolo():
    return YOLO('final.pt')

@st.cache_resource
def get_queue():
    return queue.Queue()

model = load_yolo()
result_queue = get_queue()

if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""
if 'last_added_char' not in st.session_state:
    st.session_state['last_added_char'] = None

def clear_all():
    st.session_state['sentence'] = ""
    st.session_state['last_added_char'] = None
    while not result_queue.empty():
        try: result_queue.get_nowait()
        except queue.Empty: break

# 3. Logic Engine
class VideoTransformer(VideoTransformerBase):
    def __init__(self, threshold):
        self.last_detected = None
        self.counter = 0
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5, verbose=False)
        annotated_img = results[0].plot()
        
        if len(results[0].boxes) > 0:
            char_idx = int(results[0].boxes.cls[0])
            char_name = results[0].names[char_idx].upper().strip()
            
            if char_name == self.last_detected:
                self.counter += 1
            else:
                self.counter = 0
                self.last_detected = char_name

            if self.counter == self.threshold or self.counter == int(self.threshold * 2.5):
                result_queue.put(char_name)
                if self.counter > self.threshold * 2.5:
                    self.counter = self.threshold + 1 
        else:
            self.counter = 0
            self.last_detected = None
            result_queue.put("RESET")
                
        return annotated_img

# 4. Main UI Construction
st.markdown('<div class="header-text"><h1>🤟 Sign Language Recognition System</h1><h4>University of Mosul</h4><h5>Computer Engineering Department</h5></div>', unsafe_allow_html=True)
st.write("---")

# Layout: Left for Controls, Right for Execution
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Settings")
    speed_val = st.slider("Sensitivity", 5, 60, 20)
    st.write("") # Spacer
    st.button("🗑️ Clear All Text", on_click=clear_all)

with col_right:
    st.subheader("Detection")
    # Start button will appear ABOVE the video due to CSS
    webrtc_ctx = webrtc_streamer(
        key="uom-final-v10", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# 5. Result Display
footer_placeholder = st.empty()

while True:
    try:
        new_char = result_queue.get(timeout=0.1)
        if new_char == "RESET":
            st.session_state['last_added_char'] = None
        elif new_char in ['SPACE', 'S P A C E']:
            st.session_state['sentence'] += " "
        elif new_char in ['DELETE', 'DEL', 'DELET']:
            st.session_state['sentence'] = st.session_state['sentence'][:-1]
        else:
            st.session_state['sentence'] += new_char
    except queue.Empty:
        pass
    
    final_text = st.session_state['sentence'] if st.session_state['sentence'] else "READY..."
    footer_placeholder.markdown(f'<div class="fixed-footer">{final_text}</div>', unsafe_allow_html=True)
    
    time.sleep(0.05)
    if not webrtc_ctx.state.playing and not st.session_state['sentence']:
        break
