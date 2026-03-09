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
    file_name = "Data_2Class_V1.csv" 
    
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