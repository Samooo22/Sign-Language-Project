import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import queue
import time

# 1. Page Configuration
st.set_page_config(page_title="SLR System - University of Mosul", layout="wide")

# Optimized CSS for Perfect Alignment and Adaptive Contrast
st.markdown("""
    <style>
    /* 1. Force Start button to the TOP */
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        display: flex;
        flex-direction: column-reverse;
    }
    
    /* 2. Small Professional Camera Side-View */
    .stVideo {
        max-width: 260px !important;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        margin-top: 5px;
    }
    
    /* 3. Header Styling */
    .header-text {
        text-align: center;
        margin-bottom: 20px;
    }

    /* 4. FIXED FOOTER (Now at the absolute bottom 0) */
    .fixed-footer {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background-color: #000000; 
        color: #00FF41; 
        text-align: center; 
        padding: 15px;
        font-size: 26px; 
        font-weight: bold;
        z-index: 1000; 
        border-top: 3px solid #3b82f6;
    }

    /* 5. CREDITS SECTION (With bottom padding to clear the footer) */
    .credits-box {
        text-align: center;
        padding: 40px 0 150px 0; /* Extra padding to ensure credits are seen above footer */
        font-family: 'Segoe UI', sans-serif;
    }
    .label-text {
        color: #94a3b8;
        font-size: 14px;
        font-weight: bold;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }
    .names-text {
        color: #3b82f6;
        font-weight: 800;
        font-size: 22px;
        margin-bottom: 15px;
    }
    .supervisor-name {
        color: #3b82f6;
        font-weight: bold;
        font-size: 19px;
    }

    .stButton>button {
        width: 100%;
        background-color: #DC2626;
        color: white;
        border-radius: 6px;
        height: 48px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Model & Backend Initialization
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

# 3. Vision Engine (Repeat Detection Logic)
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

# Columns: Settings on Left | Detection on Right
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Settings")
    speed_val = st.slider("Sensitivity", 5, 60, 20)
    st.write("") 
    st.button("🗑️ Clear All Text", on_click=clear_all)

with col_right:
    st.subheader("Detection")
    # Camera Start is at TOP due to CSS reverse-flex
    webrtc_ctx = webrtc_streamer(
        key="uom-final-v14", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# 5. Output Footer (Placeholder)
footer_placeholder = st.empty()

# 6. Static Credits Section
st.write("---")
st.markdown(f"""
    <div class="credits-box">
        <p class="label-text">BY</p>
        <p class="names-text">Ismail Riyadh Ismail<br>Hayder Laith Salim</p>
        <br>
        <p class="label-text">SUPERVISOR</p>
        <p class="supervisor-name">Asst. Lect. Hiba Dhiya Ali</p>
    </div>
    """, unsafe_allow_html=True)

# 7. Real-time Output Logic
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
    
    current_out = st.session_state['sentence'] if st.session_state['sentence'] else "READY..."
    footer_placeholder.markdown(f'<div class="fixed-footer">{current_out}</div>', unsafe_allow_html=True)
    
    time.sleep(0.05)
    if not webrtc_ctx.state.playing and not st.session_state['sentence']:
        break
