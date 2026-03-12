import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import queue
import time

# 1. تحميل الموديل مرة واحدة فقط (كاش)
@st.cache_resource
def load_yolo():
    # تأكد أن ملف final.pt موجود في نفس مجلد السكربت
    return YOLO('final.pt')

# 2. إنشاء الطابور المشترك مرة واحدة لضمان تزامن البيانات
@st.cache_resource
def get_queue():
    return queue.Queue()

model = load_yolo()
result_queue = get_queue()

# إعدادات واجهة الموقع
st.set_page_config(page_title="Gestures to Phrases - UoM", layout="wide")
st.title("🤟 نظام ترجمة لغة الإشارة - جامعة الموصل")
st.markdown("### كلية الهندسة - قسم هندسة الحاسوب")
st.write("---")

# تهيئة ذاكرة الجلسة لحفظ النص وحالة التكرار
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""
if 'last_added_char' not in st.session_state:
    st.session_state['last_added_char'] = None

# 3. كلاس معالجة الفيديو (الخيط الخلفي)
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_detected = None
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # كشف الإشارات (conf=0.5 لضمان دقة عالية)
        results = model(img, conf=0.5, verbose=False)
        annotated_img = results[0].plot()
        
        if len(results[0].boxes) > 0:
            char_idx = int(results[0].boxes.cls[0])
            char_name = results[0].names[char_idx].lower().strip()
            
            # منطق الثبات (العد لـ 7 إطارات قبل الإرسال)
            if char_name == self.last_detected:
                self.counter += 1
            else:
                self.counter = 0
                self.last_detected = char_name

            if self.counter == 7:
                print(f"✅ تم التقاط الحرف: {char_name}")
                result_queue.put(char_name)
                self.counter = 0
                
        return annotated_img

# 4. تشغيل الكاميرا في الموقع
webrtc_ctx = webrtc_streamer(
    key="uom-sign-final", 
    video_transformer_factory=VideoTransformer,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 5. منطقة عرض النتائج والتحديث التلقائي
st.subheader("📝 النص المترجم حياً:")
text_display = st.empty()

# محرك المراقبة والتحديث المستمر
if webrtc_ctx.state.playing:
    while webrtc_ctx.state.playing:
        try:
            # جلب الحرف المكتشف من الطابور
            new_char = result_queue.get(timeout=0.1)
            
            # --- منطق الفلترة ومنع التكرار ---
            if new_char == 'space':
                st.session_state['sentence'] += " "
                st.session_state['last_added_char'] = " " # تحديث لمنع تكرار المسافة
            
            elif new_char in ['delet', 'del', 'delete']:
                st.session_state['sentence'] = st.session_state['sentence'][:-1]
                st.session_state['last_added_char'] = None # تصفير الذاكرة للسماح بكتابة نفس الحرف المحذوف فوراً
            
            # إضافة الحرف فقط إذا لم يكن مكرراً بشكل متتالي
            elif new_char != st.session_state.get('last_added_char'):
                st.session_state['sentence'] += new_char
                st.session_state['last_added_char'] = new_char
            
            # تحديث النص المعروض في الواجهة
            text_display.info(st.session_state['sentence'] if st.session_state['sentence'] else "ابدأ الإشارة الآن...")
            
        except queue.Empty:
            # إذا كان الطابور فارغاً، يستمر في عرض النص الحالي
            text_display.info(st.session_state['sentence'] if st.session_state['sentence'] else "بانتظار الإشارة...")
            
        # توقف طفيف جداً لضمان سلاسة العرض
        time.sleep(0.05)
else:
    text_display.warning("الرجاء الضغط على Start لبدء عملية الترجمة")

# زر مسح النص
if st.button("تفريغ النص بالكامل"):
    st.session_state['sentence'] = ""
    st.session_state['last_added_char'] = None
    st.rerun()

st.sidebar.info("هذا النظام مدعوم بتقنية YOLOv8 ومطور لصالح جامعة الموصل.")