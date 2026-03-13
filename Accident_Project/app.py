import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import urllib.request

# ==========================================
# 1. ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ (Matplotlib & Seaborn)
# ==========================================
font_path = "Sarabun-Regular.ttf"
# ถ้ายังไม่มีไฟล์ฟอนต์ในเครื่อง/เซิร์ฟเวอร์ ให้ดาวน์โหลดอัตโนมัติ
if not os.path.exists(font_path):
    url = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    urllib.request.urlretrieve(url, font_path)

# บังคับให้กราฟใช้ฟอนต์ Sarabun
fm.fontManager.addfont(font_path)
mpl.rc('font', family='Sarabun')
mpl.rcParams['axes.unicode_minus'] = False # แก้ปัญหาสัญลักษณ์ลบ (-) กลายเป็นสี่เหลี่ยม

# ==========================================
# 2. ตั้งค่าหน้าเว็บและดีไซน์ (Page Config & CSS)
# ==========================================
st.set_page_config(page_title="Road Safety Dashboard", page_icon="🚑", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Sarabun', sans-serif !important; }
        h1, h2, h3 { font-weight: 700 !important; color: #1E3A8A !important; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ฟังก์ชันโหลดข้อมูลและโมเดล
# ==========================================
@st.cache_data
def load_data():
    # 💡 ให้ Python หาที่อยู่ปัจจุบันของไฟล์ app.py บนเครื่อง
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # รายชื่อไฟล์ที่ระบบจะลองค้นหา
    possible_filenames = ['Data_2Class_V1.csv', 'Data_2Class_V1.csv.csv', 'Data_2Class_V1']
    
    for filename in possible_filenames:
        file_path = os.path.join(current_dir, filename)
        
        if os.path.exists(file_path):
            try:
                # ลองอ่านแบบมาตรฐาน (utf-8-sig)
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                # คลีนข้อมูลพิกัด
                if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
                    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
                    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                    
                # สร้างคอลัมน์ระดับความเสี่ยง (ถ้ามี code)
                if 'code_ระดับความเสี่ยง' in df.columns:
                    df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
                    
                return df
            
            except UnicodeDecodeError:
                try:
                    # ถ้าอ่านไม่ได้ ลองอ่านแบบภาษาไทย Windows (windows-874)
                    df = pd.read_csv(file_path, encoding='windows-874')
                    
                    if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
                        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
                        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                        
                    if 'code_ระดับความเสี่ยง' in df.columns:
                        df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
                    return df
                except Exception as e:
                    st.error(f"🚨 ไฟล์มีปัญหาเรื่องภาษาไทย: {e}")
                    return None
            except Exception as e:
                st.error(f"🚨 อ่านไฟล์ไม่ได้: {e}")
                return None
                
    return None

@st.cache_resource
def load_ml_assets():
    # ระบบค้นหาไฟล์โมเดลอัตโนมัติ 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(current_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        feature_cols = joblib.load(os.path.join(current_dir, 'feature_columns.pkl'))
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, None

df = load_data()
model, scaler, feature_cols = load_ml_assets()

# ==========================================
# 4. ระบบ Login ใน Sidebar
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204003.png", width=100)
    st.title("ระบบเจ้าหน้าที่")
    
    if not st.session_state['logged_in']:
        if not st.session_state['show_login']:
            st.info("กรุณาล็อกอินเพื่อจัดการข้อมูล")
            if st.button("🔐 เข้าสู่ระบบ (Login)", use_container_width=True, type="primary"):
                st.session_state['show_login'] = True
                st.rerun()
        else:
            with st.container():
                st.markdown("---")
                user = st.text_input("ชื่อผู้ใช้งาน")
                pw = st.text_input("รหัสผ่าน", type="password")
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    if st.button("ยืนยัน", use_container_width=True, type="primary"):
                        if user == "admin" and pw == "admin123":
                            st.session_state['logged_in'] = True
                            st.session_state['show_login'] = False
                            st.success("สำเร็จ!")
                            st.rerun()
                        else:
                            st.error("ข้อมูลผิดพลาด!")
                with col_l2:
                    if st.button("ยกเลิก", use_container_width=True):
                        st.session_state['show_login'] = False
                        st.rerun()
    else:
        st.success("✅ สถานะ: ผู้ดูแลระบบ")
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

# ==========================================
# 5. ส่วนแสดงผลเนื้อหาหลัก (Tabs)
# ==========================================
st.title("🚑 ระบบวิเคราะห์ความรุนแรงอุบัติเหตุทางถนน")
st.subheader("เขตสุขภาพที่ 11 | โครงงานพัฒนาโมเดล Machine Learning")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 สถิติภาพรวม", 
    "🗺️ แผนที่จุดเสี่ยง", 
    "🚨 ระบบทำนายความรุนแรง", 
    "📝 จัดการข้อมูล (CRUD)"
])

# ------------------------------------------
# TAB 1: สถิติ (Overview)
# ------------------------------------------
with tab1:
    if df is not None:
        st.markdown("### 📊 ภาพรวมสถิติอุบัติเหตุทางถนน")
        
        # --- ส่วนที่ 1: กล่องตัวเลขสรุป (KPI Metrics) ---
        col1, col2, col3, col4 = st.columns(4)
        
        total_acc = len(df)
        high_risk = len(df[df['ระดับความเสี่ยง'] == 'เสี่ยงสูง']) if 'ระดับความเสี่ยง' in df.columns else 0
        low_risk = len(df[df['ระดับความเสี่ยง'] == 'เสี่ยงต่ำ']) if 'ระดับความเสี่ยง' in df.columns else 0
        total_dead = int(df['ผู้เสียชีวิต'].sum()) if 'ผู้เสียชีวิต' in df.columns else 0
        
        col1.metric("🚨 จำนวนอุบัติเหตุรวม", f"{total_acc:,} ครั้ง")
        col2.metric("🔴 เสี่ยงสูง (High Risk)", f"{high_risk:,} ครั้ง")
        col3.metric("🟢 เสี่ยงต่ำ (Low Risk)", f"{low_risk:,} ครั้ง")
        col4.metric("💀 ผู้เสียชีวิตรวม", f"{total_dead:,} คน")
        
        st.markdown("---")
        
        # --- ส่วนที่ 2: กราฟ (Charts) ---
        st.markdown("#### 📈 วิเคราะห์ปัจจัยการเกิดอุบัติเหตุ")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("**สัดส่วนระดับความเสี่ยง**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            if 'ระดับความเสี่ยง' in df.columns:
                sns.countplot(data=df, x='ระดับความเสี่ยง', palette=['#FF2B2B', '#09AB3B'], ax=ax1, order=['เสี่ยงสูง', 'เสี่ยงต่ำ'])
                ax1.set_ylabel("จำนวน (ครั้ง)")
                ax1.set_xlabel("")
                st.pyplot(fig1)
            else:
                st.info("ไม่พบคอลัมน์ 'ระดับความเสี่ยง'")
            
        with col_g2:
            st.write("**ช่วงเวลาที่เกิดอุบัติเหตุบ่อยที่สุด**")
            if 'ช่วงเวลา' in df.columns:
                counts = df['ช่วงเวลา'].value_counts()
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                pal = sns.color_palette("Blues_r", len(counts))
                sns.barplot(x=counts.index, y=counts.values, palette=pal, ax=ax2)
                ax2.set_ylabel("จำนวน (ครั้ง)")
                ax2.set_xlabel("")
                st.pyplot(fig2)
            else:
                st.info("ไม่พบคอลัมน์ 'ช่วงเวลา'")
                
        st.markdown("---")
        
        # --- ส่วนที่ 3: ตารางข้อมูลดิบ ---
        st.markdown("#### 📋 ข้อมูลรายละเอียด (Raw Data)")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"💡 กำลังแสดงผล 100 รายการแรก จากข้อมูลทั้งหมด {total_acc:,} รายการ")
        
    else:
        st.error("⚠️ ไม่พบไฟล์ข้อมูล CSV กรุณาตรวจสอบการอัปโหลด")

# ------------------------------------------
# TAB 2: แผนที่ (Map)
# ------------------------------------------
with tab2:
    if df is not None:
        st.subheader("จุดเกิดเหตุอุบัติเหตุในพื้นที่")
        map_df = df[['LATITUDE', 'LONGITUDE']].rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
        st.map(map_df)
    else:
        st.info("ไม่มีข้อมูลพิกัดเพื่อแสดงผล")

# ------------------------------------------
# TAB 3: ทำนายผล (Prediction)
# ------------------------------------------
with tab3:
    if not st.session_state['logged_in']:
        st.warning("🔒 เนื้อหาส่วนนี้เฉพาะเจ้าหน้าที่ กรุณาล็อกอินที่แถบด้านข้าง")
    elif model is None:
        st.error("🚨 ไม่พบไฟล์โมเดล AI (.pkl) กรุณาตรวจสอบว่าอัปโหลดไฟล์โมเดลแล้วหรือไม่")
    else:
        col_in, col_res = st.columns([1, 1])
        with col_in:
            st.write("### 📝 ระบุรายละเอียดอุบัติเหตุ")
            with st.form("ml_form"):
                time_val = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                weather_val = st.selectbox("สภาพอากาศ", ["แจ่มใส", "ฝนตก", "หมอกทึบ", "ไม่ระบุ"])
                mc_val = st.number_input("รถจักรยานยนต์ (คัน)", 0, 10, 1)
                submit_pred = st.form_submit_button("วิเคราะห์ความรุนแรง 🔍")
        
        with col_res:
            st.write("### 📊 ผลการทำนาย")
            if submit_pred:
                st.success("✅ โมเดลเชื่อมต่อสำเร็จ! (กำลังรอการปรับแต่ง Features ให้ตรงกับโมเดลจริง)")

# ------------------------------------------
# TAB 4: จัดการข้อมูล (CRUD)
# ------------------------------------------
with tab4:
    if not st.session_state['logged_in']:
        st.warning("🔒 กรุณาเข้าสู่ระบบเพื่อเข้าถึงฐานข้อมูล")
    else:
        st.write("### 🗃️ ฐานข้อมูลอุบัติเหตุ (CRUD Management)")
        
        if df is not None:
            # ค้นหาและแสดงผล
            search = st.text_input("🔍 ค้นหาข้อมูล (จังหวัด, ช่วงเวลา, ฯลฯ)")
            if search:
                filtered_df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"แสดงข้อมูล 100 รายการล่าสุด จากทั้งหมด {len(df):,} รายการ")
            
            st.markdown("---")
            
            # ฟอร์มเพิ่ม/แก้ไข/ลบ
            col_c, col_ud = st.columns(2)
            
            with col_c:
                st.write("#### ➕ เพิ่มข้อมูลใหม่")
                with st.form("add_form"):
                    new_prov = st.text_input("จังหวัด")
                    new_time = st.selectbox("ช่วงเวลาเกิดเหตุ", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                    if st.form_submit_button("บันทึกข้อมูล"):
                        st.toast("บันทึกข้อมูลสำเร็จ! (โหมดจำลอง)")
            
            with col_ud:
                st.write("#### ✏️ แก้ไข หรือ ลบข้อมูล")
                idx_to_edit = st.number_input("ระบุลำดับ (Index)", 0, len(df)-1 if len(df)>0 else 0, 0)
                if len(df) > 0:
                    st.write("**ข้อมูลที่เลือก:**", df.iloc[idx_to_edit][['จังหวัด', 'ช่วงเวลา']].to_dict() if 'จังหวัด' in df.columns else "ไม่มีข้อมูลจังหวัด")
                
                c_edit, c_del = st.columns(2)
                with c_edit:
                    if st.button("🔄 อัปเดตข้อมูล", use_container_width=True):
                        st.info(f"อัปเดตข้อมูลลำดับที่ {idx_to_edit} แล้ว")
                with c_del:
                    if st.button("🗑️ ลบข้อมูลนี้", use_container_width=True, type="primary"):
                        st.error(f"ลบข้อมูลลำดับที่ {idx_to_edit} แล้ว")
        else:
            st.error("ไม่สามารถจัดการข้อมูลได้เนื่องจากไม่มีข้อมูล")