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

# CSS - Perfectly Aligned Layout
st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        display: flex; flex-direction: column-reverse; align-items: center;
    }
    .stVideo {
        max-width: 480px !important; 
        border: 3px solid #3b82f6; border-radius: 15px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
    }

    .translation-box {
        background-color: #000000; 
        color: #00FF41; 
        text-align: center; 
        padding: 15px;
        font-size: 26px; 
        font-weight: bold;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        margin: 10px 0;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        box-sizing: border-box;
        word-break: keep-all; 
        overflow-wrap: break-word;
        white-space: normal;
        line-height: 1.4;
    }

    .header-text { text-align: center; margin-bottom: 25px; }
    
    .credits-bottom-box {
        text-align: center; padding: 40px 0 50px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .univ-text { color: #94a3b8; font-weight: bold; font-size: 14px; letter-spacing: 3px; }
    .names-text { color: #3b82f6; font-weight: 800; font-size: 24px; margin-top: 10px; }
    .supervisor-text { color: #ffffff; font-weight: bold; font-size: 18px; margin-top: 5px; }
    
    .stButton>button {
        width: 100%; background-color: #DC2626; color: white;
        border-radius: 8px; height: 45px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Dictionary Database
FULL_DICTIONARY = [
    "APPLE", "ABOUT", "AFTER", "ALWAYS", "AND", "BABY", "BALL", "BECAUSE", "BIG", "BOOK",
    "CAN", "CAR", "CLEAN", "COME", "CAT", "DAD", "DAY", "DID", "DIFFERENT", "DO", "DRINK",
    "EAT", "EVERY", "EGG", "EYE", "EAR", "FAMILY", "FATHER", "FISH", "FOOD", "FOR", "FRIEND",
    "GIVE", "GO", "GOOD", "GREAT", "GREEN", "HELLO", "HELP", "HOME", "HOW", "HAVE", "HAPPY",
    "ICE", "IF", "IMPORTANT", "IN", "IS", "IT", "JUMP", "JOB", "JUST", "JUICE", "JOY",
    "KEEP", "KIND", "KNOW", "KEY", "KITCHEN", "LIKE", "LITTLE", "LOOK", "LOVE", "LEARN",
    "MAKE", "ME", "MOM", "MORE", "MY", "MOSUL", "NAME", "NEW", "NICE", "NIGHT", "NO", "NOT",
    "OFF", "OLD", "ON", "ONE", "OPEN", "OTHER", "PAPER", "PEOPLE", "PLACE", "PLAY", "PLEASE",
    "QUIET", "QUESTION", "QUICK", "QUEEN", "READ", "READY", "RED", "RIGHT", "RUN",
    "SCHOOL", "SHE", "SIGN", "SOME", "STOP", "THANK", "THAT", "THE", "THINK", "TIME", "TO",
    "UNIVERSITY", "UP", "US", "USE", "UNDER", "VERY", "VISIT", "VOICE", "VEGETABLE",
    "WANT", "WATER", "WE", "WELCOME", "WITH", "WORK", "XRAY", "XYLO", "YES", "YOU", "YOUR", "YELLOW", "ZERO", "ZOO"
]

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
    if parts:
        parts[-1] = word
        st.session_state['sentence'] = " ".join(parts) + " "
    else:
        st.session_state['sentence'] = word + " "

# 3. Vision Processor
class VideoTransformer(VideoTransformerBase):
    def __init__(self, threshold):
        self.last_detected = None
        self.counter = 0
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.35, verbose=False)
        annotated_img = results[0].plot()
        if len(results[0].boxes) > 0:
            char_idx = int(results[0].boxes.cls[0])
            char_name = results[0].names[char_idx].upper().strip()
            if char_name == self.last_detected: self.counter += 1
            else: self.counter = 0; self.last_detected = char_name
            
            if self.counter == self.threshold:
                result_queue.put(char_name)
            elif self.counter == self.threshold * 4:
                result_queue.put(char_name)
                self.counter = self.threshold + 1 
        else:
            self.counter = 0; self.last_detected = None; result_queue.put("RESET")
        return annotated_img

# 4. Interface Setup
st.markdown('<div class="header-text"><h1>🤟 Smart Sign Language Recognition System</h1></div>', unsafe_allow_html=True)
st.write("---")

col_left, col_mid, col_right = st.columns([1, 1.8, 2.2], gap="large")

with col_left:
    st.subheader("⚙️ Control Panel")
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
        if matches:
            for m in matches[:5]:
                st.button(f"➕ {m}", key=f"btn_{m}", on_click=complete_word, args=(m,))
        else: st.info("No matches.")
    else: st.info("Signal a letter...")

with col_mid:
    st.subheader("🎥 Intelligent Feed")
    webrtc_ctx = webrtc_streamer(
        key="uom-final-v38-optimized", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True, 
        # تقليل جودة الكاميرا لزيادة الاستقرار
        media_stream_constraints={"video": {"width": 480, "height": 360}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col_right:
    st.subheader("📖 Reference Guide")
    LOCAL_IMAGE_PATH = "asl_guide.png"
    if os.path.exists(LOCAL_IMAGE_PATH):
        try:
            img_g = Image.open(LOCAL_IMAGE_PATH)
            t_w = 200
            w_p = (t_w / float(img_g.size[0]))
            h_s = int((float(img_g.size[1]) * float(w_p)))
            img_g = img_g.resize((t_w, h_s), Image.Resampling.LANCZOS)
            st.image(img_g, caption='Fingerspelling Reference Chart', width=600)
        except Exception as e: st.error(f"Image Error: {e}")
    else: st.warning("Guide image missing.")

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
            if new_char:
                if new_char == "RESET": pass
                elif new_char in ['SPACE', 'S P A C E']: st.session_state['sentence'] += " "
                elif new_char in ['DELETE', 'DEL', 'DELET']: st.session_state['sentence'] = st.session_state['sentence'][:-1]
                else: st.session_state['sentence'] += new_char
                st.rerun()
        except queue.Empty: pass
        
        display_text = st.session_state['sentence'] if st.session_state['sentence'] else "READY..."
        output_placeholder.markdown(f'<div class="translation-box">{display_text}</div>', unsafe_allow_html=True)
        
        time.sleep(0.05)
        if not webrtc_ctx.state.playing: break
