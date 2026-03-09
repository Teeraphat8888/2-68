import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Config & CSS)
# ==========================================
st.set_page_config(page_title="Road Safety Dashboard - Health Region 11", page_icon="🚑", layout="wide")

# เพิ่ม CSS เพื่อให้แสดงผลภาษาไทย (Sarabun) ได้สวยงาม
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Sarabun', sans-serif !important; }
        h1, h2, h3 { font-weight: 700 !important; color: #1E3A8A !important; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
        .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ฟังก์ชันโหลดข้อมูล (Data & ML Assets)
# ==========================================
@st.cache_data
def load_data():
    # ตรวจสอบชื่อไฟล์ให้ตรงกับบน GitHub (Case-sensitive)
    file_name = "Accident_Project/Data_2Class_V1.csv" 
    
    if os.path.exists(file_name):
        try:
            # ใช้ encoding='utf-8-sig' เพื่อรองรับภาษาไทยจาก Excel CSV
            df = pd.read_csv(file_name, encoding='utf-8-sig')
            
            # จัดการข้อมูลพิกัดและระดับความเสี่ยง
            if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
                df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
                df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                
            if 'code_ระดับความเสี่ยง' in df.columns:
                df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
            return df
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
            return None
    return None

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, scaler, feature_cols
    except:
        return None, None, None

# โหลดข้อมูลเข้าสู่โปรแกรม
df = load_data()
model, scaler, feature_cols = load_ml_assets()

# ==========================================
# 3. ระบบ Login (Session State)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204003.png", width=100)
    st.title("เมนูระบบ")
    
    if not st.session_state['logged_in']:
        if not st.session_state['show_login']:
            st.info("กรุณาเข้าสู่ระบบเพื่อจัดการข้อมูล")
            if st.button("🔐 เข้าสู่ระบบ (Login)", use_container_width=True, type="primary"):
                st.session_state['show_login'] = True
                st.rerun()
        else:
            st.markdown("### 🔑 ลงชื่อเข้าใช้")
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                if st.button("ตกลง", use_container_width=True, type="primary"):
                    if user == "admin" and pw == "admin123":
                        st.session_state['logged_in'] = True
                        st.session_state['show_login'] = False
                        st.rerun()
                    else:
                        st.error("ข้อมูลผิดพลาด")
            with col_l2:
                if st.button("ยกเลิก", use_container_width=True):
                    st.session_state['show_login'] = False
                    st.rerun()
    else:
        st.success("✅ เข้าสู่ระบบแล้ว (Admin)")
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

# ==========================================
# 4. ส่วนแสดงผลหลัก (Main UI)
# ==========================================
st.title("🚑 ระบบวิเคราะห์ความรุนแรงอุบัติเหตุทางถนน")
st.subheader("เขตสุขภาพที่ 11 | โดย ธีรภัทร กมลดี และ อัศนัย ราชรักษ์")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 สถิติ", "🗺️ แผนที่", "🤖 ทำนายผล", "⚙️ จัดการข้อมูล"])

# --- TAB 1: สถิติ ---
with tab1:
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนอุบัติเหตุทั้งหมด", f"{len(df):,} ครั้ง")
        if 'ระดับความเสี่ยง' in df.columns:
            c2.metric("เคสเสี่ยงสูง 🔴", f"{len(df[df['ระดับความเสี่ยง']=='เสี่ยงสูง']):,} ครั้ง")
            c3.metric("เคสเสี่ยงต่ำ 🟢", f"{len(df[df['ระดับความเสี่ยง']=='เสี่ยงต่ำ']):,} ครั้ง")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**สัดส่วนความรุนแรง**")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='ระดับความเสี่ยง', palette=['#28B463', '#D62728'], ax=ax)
            st.pyplot(fig)
        with col_g2:
            st.write("**จำนวนตามช่วงเวลา (ไล่สีจากบนลงล่าง)**")
            if 'ช่วงเวลา' in df.columns:
                counts = df['ช่วงเวลา'].value_counts()
                fig2, ax2 = plt.subplots()
                # สร้าง Palette สี Blues ไล่จากเข้มไปอ่อน
                pal = sns.color_palette("Blues_r", len(counts))
                sns.barplot(y=counts.index, x=counts.values, palette=pal, ax=ax2)
                st.pyplot(fig2)
    else:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูล 'Data_2Class_V1.csv' กรุณาตรวจสอบการอัปโหลดบน GitHub")

# --- TAB 2: แผนที่ ---
with tab2:
    if df is not None:
        st.map(df[['LATITUDE', 'LONGITUDE']].rename(columns={'LATITUDE':'lat', 'LONGITUDE':'lon'}))
    else:
        st.info("ไม่สามารถแสดงแผนที่ได้เนื่องจากไม่มีข้อมูล")

# --- TAB 3: ทำนายผล ---
with tab3:
    if not st.session_state['logged_in']:
        st.warning("🔒 กรุณาเข้าสู่ระบบที่แถบด้านข้างเพื่อใช้งานระบบทำนาย")
    elif model is None:
        st.error("🚨 ไม่พบไฟล์โมเดล (.pkl)")
    else:
        with st.form("predict_form"):
            st.write("### กรอกข้อมูลเพื่อทำนายความรุนแรง")
            time = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
            weather = st.selectbox("สภาพอากาศ", ["แจ่มใส", "ฝนตก", "หมอก", "ไม่ระบุ"])
            m_cycle = st.number_input("รถจักรยานยนต์ (คัน)", 0, 10, 1)
            submit = st.form_submit_button("วิเคราะห์ผล")
            
            if submit:
                # ส่วนนี้ต้องปรับให้ตรงกับ Feature ของโมเดลคุณ
                st.info("ระบบกำลังประมวลผล... (ผลลัพธ์จะแสดงตามโมเดลที่เทรนไว้)")

# --- TAB 4: CRUD ---
with tab4:
    if st.session_state['logged_in']:
        st.write("### 📝 จัดการฐานข้อมูล")
        if df is not None:
            st.dataframe(df.head(20))
            st.button("เพิ่มข้อมูลใหม่ (จำลอง)")
            st.button("ลบข้อมูล (จำลอง)", type="secondary")
    else:
        st.warning("🔒 เฉพาะเจ้าหน้าที่เท่านั้นที่สามารถเข้าถึงส่วนนี้ได้")