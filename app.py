import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import queue
import time
import os
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Gestures to Phrases - UoM", layout="wide")

# Persistent State
if 'sentence' not in st.session_state: st.session_state['sentence'] = ""

# CSS - Responsive Custom Header, Stability & New Credits Styling
st.markdown("""
    <style>
    /* 1. تنسيق حاوية النصوص المركزية (العنوان) */
    .text-block-center {
        text-align: center;
        width: 100%;
    }

    .univ-col-line {
        font-size: 16px;
        color: #ffffff;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .dept-line {
        font-size: 14px;
        color: #94a3b8;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    
    .hero-phrase {
        font-size: 44px;
        font-weight: 800;
        color: #3b82f6;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }

    /* 2. تنسيق قسم الاعتمادات السفلي (Credits) */
    .credits-section {
        text-align: center;
        padding: 50px 0 60px 0;
        border-top: 1px solid #333;
        margin-top: 50px;
    }
    .credits-label {
        color: #94a3b8;
        font-weight: bold;
        font-size: 12px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .credits-names {
        color: #3b82f6;
        font-weight: 800;
        font-size: 24px;
        margin-bottom: 25px;
    }
    .credits-supervisor {
        color: #ffffff;
        font-weight: bold;
        font-size: 20px;
    }

    /* التجاوب للموبايل */
    @media (max-width: 768px) {
        .univ-col-line { font-size: 12px !important; }
        .dept-line { font-size: 11px !important; }
        .hero-phrase { font-size: 24px !important; letter-spacing: 1px; }
        .credits-names { font-size: 18px !important; }
        .credits-supervisor { font-size: 16px !important; }
    }

    /* تثبيت أبعاد الكاميرا */
    [data-testid="stHorizontalBlock"] { align-items: flex-start !important; }
    .stVideo {
        width: 480px !important; height: 360px !important;
        aspect-ratio: 4 / 3 !important; object-fit: cover !important;
        border: 4px solid #3b82f6; border-radius: 15px; overflow: hidden;
    }

    .translation-box {
        background-color: #000000; color: #00FF41; text-align: center; 
        padding: 15px; font-size: 26px; font-weight: bold;
        border: 2px solid #3b82f6; border-radius: 10px;
        min-height: 110px; width: 100%; box-sizing: border-box;
    }
    
    .stButton>button { width: 100%; background-color: #DC2626; color: white; border-radius: 8px; height: 45px; font-weight: bold; }
    div[data-testid="stPopover"] > button {
        background-color: #3b82f6 !important; color: white !important;
        border-radius: 8px !important; height: 45px !important; font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Resources
FULL_DICTIONARY = ["APPLE", "ABOUT", "AFTER", "ALWAYS", "AND", "HELLO", "HELP", "NAME", "MOSUL", "UNIVERSITY", "COMPUTER", "ENGINEERING", "THANK", "YOU", "YES", "NO"]

@st.cache_resource
def load_yolo(): return YOLO('final.pt')
@st.cache_resource
def get_queue(): return queue.Queue()

model = load_yolo()
result_queue = get_queue()

def clear_all():
    st.session_state['sentence'] = ""
    while not result_queue.empty():
        try: result_queue.get_nowait()
        except queue.Empty: break

def complete_word(word):
    parts = st.session_state['sentence'].split()
    if parts: parts[-1] = word; st.session_state['sentence'] = " ".join(parts) + " "
    else: st.session_state['sentence'] = word + " "

# 3. Vision Processor
class VideoTransformer(VideoTransformerBase):
    def __init__(self, threshold):
        self.last_detected = None
        self.counter = 0
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(cv2.resize(img, (320, 320)), conf=0.35, verbose=False)
        annotated_img = results[0].plot() if len(results[0].boxes) > 0 else img
        if len(results[0].boxes) > 0:
            char_idx = int(results[0].boxes.cls[0])
            char_name = results[0].names[char_idx].upper().strip()
            if char_name == self.last_detected: self.counter += 1
            else: self.counter = 0; self.last_detected = char_name
            if self.counter == self.threshold: result_queue.put(char_name)
        else: self.counter = 0; self.last_detected = None; result_queue.put("RESET")
        return cv2.resize(annotated_img, (640, 480))

# 4. Interface Setup (Header)
col_logo_left, col_text, col_logo_right = st.columns([1, 4, 1])

with col_logo_left:
    if os.path.exists("col.png"): st.image("col.png", width=120)
    else: st.info("Col Logo")

with col_text:
    st.markdown("""
        <div class="text-block-center">
            <p class="univ-col-line">UNIVERSITY OF MOSUL • COLLEGE OF ENGINEERING</p>
            <p class="dept-line">COMPUTER ENGINEERING DEPARTMENT</p>
            <h1 class="hero-phrase">🤟 From Gesture to Phrase</h1>
        </div>
        """, unsafe_allow_html=True)

with col_logo_right:
    if os.path.exists("dep.png"): st.image("dep.png", width=120)
    else: st.info("Dep Logo")

st.write("---")

# Main Columns
col_left, col_mid = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("⚙️ Control Panel")
    with st.popover("📖 Open Signs Guide"):
        if os.path.exists("asl_guide.png"): st.image("asl_guide.png", use_container_width=True)
    
    speed_val = st.slider("Detection Delay", 3, 100, 15)
    st.button("🗑️ Clear Translation", on_click=clear_all)
    st.write("📝 **Live Translation:**")
    output_placeholder = st.empty()
    display_text = st.session_state['sentence'] if st.session_state['sentence'] else "READY..."
    output_placeholder.markdown(f'<div class="translation-box">{display_text}</div>', unsafe_allow_html=True)
    
    st.write("---")
    st.write("🔍 **Smart Suggestions:**")
    cur_s = st.session_state['sentence']
    last_word_seg = cur_s.split()[-1] if cur_s.strip() and not cur_s.endswith(" ") else ""
    if last_word_seg:
        matches = [w for w in FULL_DICTIONARY if w.startswith(last_word_seg.upper()) and w != last_word_seg.upper()]
        for m in matches[:5]: st.button(f"➕ {m}", key=f"btn_{m}", on_click=complete_word, args=(m,))

with col_mid:
    st.subheader("🎥 Intelligent Feed")
    webrtc_ctx = webrtc_streamer(
        key="uom-presentation-v59", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True, 
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# 5. New Professional Credits Section (Bottom)
st.markdown("""
    <div class="credits-section">
        <p class="credits-label">BY</p>
        <p class="credits-names">Ismail Riyadh Ismail & Hayder Laith Salim</p>
        <p class="credits-label">SUPERVISOR</p>
        <p class="credits-supervisor">Asst. Lect. Hiba Dhiya Ali</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Runtime Engine
if webrtc_ctx.state.playing:
    while True:
        try:
            new_char = result_queue.get(timeout=0.1)
            if new_char and new_char != "RESET":
                if new_char in ["DELETE", "DEL"]:
                    if len(st.session_state['sentence']) > 0:
                        st.session_state['sentence'] = st.session_state['sentence'][:-1]
                elif new_char in ["SPACE"]:
                    st.session_state['sentence'] += " "
                else:
                    st.session_state['sentence'] += new_char
                st.rerun()
        except queue.Empty: pass
        display_text = st.session_state['sentence'] if st.session_state['sentence'] else "READY..."
        output_placeholder.markdown(f'<div class="translation-box">{display_text}</div>', unsafe_allow_html=True)
        time.sleep(0.05)
        if not webrtc_ctx.state.playing: break
