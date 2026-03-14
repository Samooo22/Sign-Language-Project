import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO
import cv2
import queue
import time
import os
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Smart SLR - Mosul University", layout="wide")

# Persistent State
if 'sentence' not in st.session_state: st.session_state['sentence'] = ""

# CSS - قفل الأبعاد بشكل صارم لمنع التمطط والارتجاج
st.markdown("""
    <style>
    /* 1. تثبيت العمود الأوسط ومنع التغيير في ارتفاعه */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }

    /* 2. قفل أبعاد الكاميرا ومنع التمطط الطولي */
    .stVideo {
        width: 480px !important;
        height: 360px !important;
        aspect-ratio: 4 / 3 !important;
        object-fit: cover !important; /* يضمن عدم تمطط الصورة داخل الإطار */
        border: 4px solid #3b82f6;
        border-radius: 15px;
        overflow: hidden;
    }

    /* 3. تثبيت مكان زر الـ Stop والحاوية */
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 420px !important; /* مساحة محجوزة مسبقاً للكاميرا والزر */
        max-height: 420px !important;
    }

    /* صندوق الترجمة الجانبي */
    .translation-box {
        background-color: #000000; color: #00FF41; text-align: center; 
        padding: 15px; font-size: 26px; font-weight: bold;
        border: 2px solid #3b82f6; border-radius: 10px;
        min-height: 120px; width: 100%; box-sizing: border-box;
        word-break: keep-all; overflow-wrap: break-word; line-height: 1.4;
    }

    .header-text { text-align: center; margin-bottom: 25px; }
    .credits-bottom-box { text-align: center; padding: 40px 0 50px 0; font-family: 'Segoe UI', sans-serif; }
    .univ-text { color: #94a3b8; font-weight: bold; font-size: 14px; letter-spacing: 3px; }
    .names-text { color: #3b82f6; font-weight: 800; font-size: 22px; margin-top: 10px; }
    .supervisor-text { color: #ffffff; font-weight: bold; font-size: 18px; margin-top: 5px; }
    
    .stButton>button { width: 100%; background-color: #DC2626; color: white; border-radius: 8px; height: 45px; font-weight: bold; }
    div[data-testid="stPopover"] > button {
        background-color: #3b82f6 !important; color: white !important;
        border-radius: 8px !important; height: 45px !important; font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Complete Dictionary Database
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
        # معالجة بحجم ثابت لضمان استقرار الخرج
        results = model(cv2.resize(img, (320, 320)), conf=0.35, verbose=False)
        annotated_img = results[0].plot() if len(results[0].boxes) > 0 else img
        
        if len(results[0].boxes) > 0:
            char_idx = int(results[0].boxes.cls[0])
            char_name = results[0].names[char_idx].upper().strip()
            if char_name == self.last_detected: self.counter += 1
            else: self.counter = 0; self.last_detected = char_name
            if self.counter == self.threshold:
                result_queue.put(char_name)
        else:
            self.counter = 0; self.last_detected = None; result_queue.put("RESET")
        
        # التأكد من أن الصورة الخارجة دائماً بنفس الحجم الأصلي للكاميرا
        return cv2.resize(annotated_img, (640, 480))

# 4. Interface Setup
st.markdown('<div class="header-text"><h1>🤟 Smart Sign Language Recognition System</h1></div>', unsafe_allow_html=True)
st.write("---")

col_left, col_mid = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("⚙️ Control Panel")
    with st.popover("📖 Open Signs Guide"):
        LOCAL_IMAGE_PATH = "asl_guide.png"
        if os.path.exists(LOCAL_IMAGE_PATH):
            img_g = Image.open(LOCAL_IMAGE_PATH)
            st.image(img_g, use_container_width=True)
    
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
        key="uom-fixed-v50", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True, 
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# 5. Official Credits Section
st.write("---")
st.markdown(f"""
    <div class="credits-bottom-box">
        <p class="univ-text">UNIVERSITY OF MOSUL • COLLEGE OF ENGINEERING</p>
        <p class="names-text">Ismail Riyadh Ismail & Hayder Laith Salim</p>
        <p class="supervisor-text">Supervised by: Asst. Lect. Hiba Dhiya Ali</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Runtime Engine
if webrtc_ctx.state.playing:
    while True:
        try:
            new_char = result_queue.get(timeout=0.1)
            if new_char and new_char != "RESET":
                st.session_state['sentence'] += new_char
                st.rerun()
        except queue.Empty: pass
        display_text = st.session_state['sentence'] if st.session_state['sentence'] else "READY..."
        output_placeholder.markdown(f'<div class="translation-box">{display_text}</div>', unsafe_allow_html=True)
        time.sleep(0.05)
        if not webrtc_ctx.state.playing: break
