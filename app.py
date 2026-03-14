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

# CSS - High Stability & Large UI
st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        display: flex; flex-direction: column-reverse; align-items: center;
    }
    .stVideo {
        max-width: 480px !important; 
        border: 4px solid #3b82f6; border-radius: 15px;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
    }
    .header-text { text-align: center; margin-bottom: 25px; }
    .fixed-footer {
        position: fixed; bottom: 0; left: 0; width: 100%;
        background-color: #000000; color: #00FF41; 
        text-align: center; padding: 22px;
        font-size: 34px; font-weight: bold;
        z-index: 2000; border-top: 5px solid #3b82f6;
    }
    .credits-bottom-box {
        text-align: center; padding: 40px 0 200px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .names-text { color: #3b82f6; font-weight: 800; font-size: 26px; }
    .stButton>button {
        width: 100%; background-color: #DC2626; color: white;
        border-radius: 10px; height: 50px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Render Footer
f_text = st.session_state['sentence'] if st.session_state['sentence'] else "SYSTEM READY..."
st.markdown(f'<div class="fixed-footer">{f_text}</div>', unsafe_allow_html=True)

# 2. Expanded Dictionary
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

# 3. Enhanced Vision Processor
class VideoTransformer(VideoTransformerBase):
    def __init__(self, threshold):
        self.last_detected = None
        self.counter = 0
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # High-Speed confidence set to 0.35
        results = model(img, conf=0.35, verbose=False)
        
        annotated_img = results[0].plot()
        if len(results[0].boxes) > 0:
            char_idx = int(results[0].boxes.cls[0])
            char_name = results[0].names[char_idx].upper().strip()
            
            if char_name == self.last_detected:
                self.counter += 1
            else:
                self.counter = 0
                self.last_detected = char_name

            # --- تحسين المنطق: جعل الفرق واضحاً جداً للمستخدم ---
            # الآن، الحساسية العالية (رقم صغير) تجعل الحرف يظهر فوراً
            # الحساسية المنخفضة (رقم كبير) تجبر المستخدم على الثبات لفترة طويلة جداً
            if self.counter == self.threshold:
                result_queue.put(char_name)
            elif self.counter == self.threshold * 4: # تأخير كبير لتكرار الحرف
                result_queue.put(char_name)
                self.counter = self.threshold + 1 
        else:
            self.counter = 0
            self.last_detected = None
            result_queue.put("RESET")
                
        return annotated_img

# 4. Balanced UI Construction
st.markdown('<div class="header-text"><h1>🤟 Smart Sign Language Recognition System</h1><h4>University of Mosul - College of Engineering</h4></div>', unsafe_allow_html=True)
st.write("---")

col_left, col_mid, col_right = st.columns([1, 2, 2.2], gap="large")

with col_left:
    st.subheader("⚙️ Control Panel")
    
    # --- زيادة مدى السلايدر لجعل التأثير واضحاً (من 3 إلى 100) ---
    # القيمة 10 هي كتابة سريعة جداً، القيمة 60 تتطلب ثباتاً طويلاً
    speed_val = st.slider("Detection Delay (Low = Fast)", 3, 100, 15)
    
    st.button("🗑️ Clear Translation", on_click=clear_all)
    st.write("---")
    st.write("🔍 **Smart Suggestions:**")
    
    cur_s = st.session_state['sentence']
    last_w = cur_s.split()[-1] if cur_s.strip() and not cur_s.endswith(" ") else ""
    
    if last_w:
        matches = [w for w in FULL_DICTIONARY if w.startswith(last_w.upper()) and w != last_w.upper()]
        if matches:
            for m in matches[:6]:
                st.button(f"➕ {m}", key=f"btn_{m}", on_click=complete_word, args=(m,))
        else: st.info("No matches.")
    else: st.info("Signal a letter...")

with col_mid:
    st.subheader("🎥 Intelligent Feed")
    webrtc_ctx = webrtc_streamer(
        key="uom-final-master-v34", 
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(threshold=speed_val),
        async_processing=True, media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col_right:
    st.subheader("📖 Reference Guide")
    L_PATH = "asl_guide.png"
    if os.path.exists(L_PATH):
        try:
            img_g = Image.open(L_PATH)
            # Increased width to 600px for maximum clarity
            t_w = 600
            w_p = (t_w / float(img_g.size[0]))
            h_s = int((float(img_g.size[1]) * float(w_p)))
            img_g = img_g.resize((t_w, h_s), Image.Resampling.LANCZOS)
            st.image(img_g, caption='ASL Fingerspelling Reference Chart', width=600)
        except Exception as e: st.error(f"Image Error: {e}")
    else: st.warning("Guide image 'asl_guide.png' missing.")

# 5. Credits
st.write("---")
st.markdown(f"""
    <div class="credits-bottom-box">
        <p style="color:#94a3b8; font-weight:bold; letter-spacing:4px;">RESEARCH TEAM</p>
        <p class="names-text">Ismail Riyadh Ismail & Hayder Laith Salim</p>
        <br>
        <p style="color:#94a3b8; font-weight:bold; letter-spacing:4px;">SUPERVISED BY</p>
        <p style="color:#3b82f6; font-weight:bold; font-size:24px;">Asst. Lect. Hiba Dhiya Ali</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Real-time Engine
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
        time.sleep(0.05)
        if not webrtc_ctx.state.playing: break
