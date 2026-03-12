import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import queue
import time

# 1. Page Configuration
st.set_page_config(page_title="SLR System - University of Mosul", layout="wide")

# Professional CSS for Centered Layout and Fixed Footer
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stApp { align-items: center; }
    
    /* Centering the container */
    [data-testid="stVerticalBlock"] {
        text-align: center;
        align-items: center;
    }
    
    /* Video Box Styling */
    .stVideo {
        max-width: 500px !important;
        border: 5px solid #1E3A8A; /* University Blue */
        border-radius: 15px;
        margin: auto;
    }
    
    /* Official Fixed Footer */
    .fixed-footer {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background-color: #0F172A; color: #10B981; /* Emerald Green */
        text-align: center; padding: 25px;
        font-size: 38px; font-weight: bold;
        z-index: 1000; border-top: 5px solid #1E3A8A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Control Panel Styling */
    .stSlider { max-width: 400px; margin: auto; }
    .stButton>button {
        background-color: #DC2626; color: white;
        border-radius: 8px; width: 200px; height: 45px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Model & Queue Loading
@st.cache_resource
def load_yolo():
    return YOLO('final.pt')

@st.cache_resource
def get_queue():
    return queue.Queue()

model = load_yolo()
result_queue = get_queue()

# Session State Initialization
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

# 3. AI Processing Class (Repeat Logic included)
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
            char_name = results[0].names[char_idx].upper().strip() # Upper case for English letters
            
            if char_name == self.last_detected:
                self.counter += 1
            else:
                self.counter = 0
                self.last_detected = char_name

            # Dynamic Threshold for speed
            if self.counter == self.threshold or self.counter == int(self.threshold * 2.5):
                result_queue.put(char_name)
                if self.counter > self.threshold * 2.5:
                    self.counter = self.threshold + 1 
        else:
            self.counter = 0
            self.last_detected = None
            result_queue.put("RESET")
                
        return annotated_img

# 4. Official UI Layout (Top-Down Flow)
st.title("🤟 Sign Language Recognition System")
st.markdown("#### University of Mosul - College of Engineering")
st.markdown("###### Computer Engineering Department")
st.write("---")

# Recognition Speed Slider
speed_val = st.slider("Recognition Sensitivity (Lower is Faster)", 5, 60, 20)

# Clear Button
if st.button("🗑️ Clear All Text", on_click=clear_all):
    st.toast("Text Cleared")

# Centered Camera Implementation
left_sp, center_col, right_sp = st.columns([1, 2, 1])

with center_col:
    webrtc_ctx = webrtc_streamer(
        key="uom-sign-official", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# 5. Continuous Output Update (Fixed Footer)
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
    
    display_text = st.session_state['sentence'] if st.session_state['sentence'] else "WAITING FOR SIGN..."
    footer_placeholder.markdown(
        f'<div class="fixed-footer">📝 TRANSLATED TEXT: {display_text}</div>', 
        unsafe_allow_html=True
    )
    
    time.sleep(0.05)
    if not webrtc_ctx.state.playing and not st.session_state['sentence']:
        break
