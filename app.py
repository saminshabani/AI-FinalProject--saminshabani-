import streamlit as st
import pandas as pd
import plotly.express as px
import time
import numpy as np
import tensorflow as tf
from PIL import Image

# --- تنظیمات صفحه ---
st.set_page_config(page_title="پنل تشخیص هوشمند", layout="wide")

# --- بارگذاری مدل (منطق کد اول) ---
@st.cache_resource
def load_trained_model():
    try:
        # نام فایل مدل را با نام ذخیره شده در نوت‌بوک ست کنید
        model = tf.keras.models.load_model('models/densenet_finetuned_model.h5')
        return model
    except Exception as e:
        return None

model = load_trained_model()
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# --- تابع پیش‌پردازش و پیش‌بینی (منطق کد اول مطابق با نوت‌بوک فاز ۲) ---
def predict_image(image, model):
    img = image.convert('RGB')
    img = img.resize((224, 224)) # ابعاد دقیق نوت‌بوک
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # برگرداندن نام کلاس و درصد اطمینان
    return class_names[class_idx], confidence

# --- استایل CSS (ترکیب دیزاین کد دوم) ---
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    * {font-family: 'Vazirmatn', sans-serif !important; direction: rtl;}
    header, footer, #MainMenu {visibility: hidden !important;}
    .stApp {background-color: #0d1117;}

    .main-title {
        text-align: center; color: white; 
        margin-top: -60px; margin-bottom: 40px;
        font-weight: bold;
    }

    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-top: 4px solid #58a6ff;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 20px;
    }

    [data-testid="stVerticalBlock"] > div:has(div.js-plotly-plot) {
        border: 2px dashed #30363d !important;
        border-radius: 15px;
        padding: 20px !important;
        background-color: #0d1117;
    }

    .custom-legend-wrap {
        display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;
        margin-top: -10px; padding-bottom: 10px; direction: ltr;
    }
    .legend-node { display: flex; align-items: center; gap: 10px; }
    .color-pill { width: 15px; height: 15px; border-radius: 4px; flex-shrink: 0; }
    .label-pill { color: #c9d1d9; font-size: 14px; white-space: nowrap; }

    .centered-section { text-align: center; display: flex; flex-direction: column; align-items: center; }
    .status-container {
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        height: 300px; background: #1c2128; border: 1px dashed #30363d; border-radius: 10px; color: #8b949e;
    }
    .result-card {
        background: #1c2128; border-right: 5px solid #238636; padding: 20px; border-radius: 8px; margin-top: 28px;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 class='main-title' style='margin-top:-80px;margin-bottom:20px;'>پنل هوشمند تشخیص پاتولوژی ریه</h2>", unsafe_allow_html=True)

tab_dash, tab_scan = st.tabs(["📊 داشبورد گزارشات", "🔍 آنالیز تصویر"])

# --- تب اول: داشبورد (کاملاً از کد دوم) ---
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    metrics = [("تصاویر X-ray", "21,165", "#49e088"), ("دقت مدل", "94.2%", "#f7d44b"), 
               ("کلاس‌ هدف", "4", "#d299ff"), ("خطای تست", "0.08", "#f85149")]
    for i, (label, val, color) in enumerate(metrics):
        with [c1, c2, c3, c4][i]:
            st.markdown(f'<div class="metric-box" style="border-top-color:{color};"><h3>{label}</h3><h2>{val}</h2></div>', unsafe_allow_html=True)

    df_stats = pd.DataFrame({
        'Label': ['Normal', 'Lung Opacity', 'COVID-19', 'Viral Pneumonia'],
        'Value': [10192, 6012, 3616, 1345]
    })
    colors_pie = ['#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
    
    fig = px.pie(df_stats, values='Value', names='Label', hole=0.6, color_discrete_sequence=colors_pie)
    fig.update_traces(textinfo='percent', hoverinfo='skip', hovertemplate=None, marker=dict(line=dict(color='#0d1117', width=3)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    legend_items_html = "".join([
        f'<div class="legend-node">'
        f'<div class="color-pill" style="background-color: {colors_pie[i]};"></div>'
        f'<span class="label-pill">{df_stats["Label"][i]}</span>'
        f'</div>'
        for i in range(len(df_stats))
    ])
    st.markdown(f'<div class="custom-legend-wrap">{legend_items_html}</div>', unsafe_allow_html=True)

# --- تب دوم: آنالیز تصویر (داشبورد کد دوم + منطق مدل کد اول) ---
with tab_scan:
    st.markdown('<div class="centered-section">', unsafe_allow_html=True)
    col_u, col_p, col_r = st.columns([1, 1, 1])
    
    with col_u:
        st.markdown("**upload**")
        up_file = st.file_uploader("تصویر رادیولوژی را بارگذاری کنید", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    with col_p:
        st.markdown("**picture**")
        if up_file:
            img_to_predict = Image.open(up_file)
            st.image(img_to_predict, use_container_width=True)
        else:
            st.markdown('<div class="status-container">تصویری بارگذاری نشده</div>', unsafe_allow_html=True)
            
    with col_r:
        st.markdown("**result**")
        if up_file:
            if model is not None:
                with st.spinner("در حال آنالیز توسط مدل فاز ۲..."):
                    # فراخوانی تابع پیش‌بینی واقعی
                    res_label, res_conf = predict_image(img_to_predict, model)
                    
                    # تعیین رنگ بر اساس نتیجه (سبز برای نرمال، قرمز برای بیماری)
                    border_color = "#238636" if res_label == "Normal" else "#f85149"
                    
                    st.markdown(f"""
                        <div class="result-card" style="border-right-color: {border_color};">
                            <h3>تشخیص: {res_label}</h3>
                            <p>اطمینان مدل: {res_conf:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("خطا: فایل best_model.h5 یافت نشد.")
        else:
            st.markdown('<div class="status-container">در انتظار آنالیز...</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: #484f58; margin-top: 50px;'>پروژه هوش مصنوعی - نسخه نهایی دمو</div>", unsafe_allow_html=True)
