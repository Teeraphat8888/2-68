import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. ตั้งค่าหน้าเว็บและดีไซน์ (Page Config & CSS)
# ==========================================
st.set_page_config(page_title="Road Safety Dashboard - Health Region 11", page_icon="🚑", layout="wide")

# CSS สำหรับปรับแต่งหน้าตาและฟอนต์ภาษาไทย
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Sarabun', sans-serif !important; }
        h1, h2, h3 { font-weight: 700 !important; color: #1E3A8A !important; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
        .stDataFrame { border: 1px solid #e6e9ef; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ฟังก์ชันหลัก (Data Loading)
# ==========================================
@st.cache_data
def load_data():
    # ตรวจสอบชื่อไฟล์ให้ตรงกับบน GitHub (แนะนำเป็น CSV เพื่อความเร็วและภาษาไทย)
    file_name = "Accident_Project/Data_2Class_V1.csv" 
    
    if os.path.exists(file_name):
        try:
            # ใช้ utf-8-sig เพื่อป้องกันภาษาไทยเพี้ยนจากไฟล์ Excel CSV
            df = pd.read_csv(file_name, encoding='utf-8-sig')
            
            # ทำความสะอาดข้อมูลเบื้องต้น
            if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
                df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
                df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                
            if 'code_ระดับความเสี่ยง' in df.columns:
                df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('Accident_Project/best_model.pkl')
        scaler = joblib.load('Accident_Project/scaler.pkl')
        feature_cols = joblib.load('Accident_Project/feature_columns.pkl')
        return model, scaler, feature_cols
    except:
        return None, None, None

df = load_data()
model, scaler, feature_cols = load_ml_assets()

# ==========================================
# 3. ระบบ Login ใน Sidebar (แบบปุ่มกดเปิด/ปิด)
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
                            st.error("ผิดพลาด!")
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
# 4. ส่วนแสดงผลเนื้อหาหลัก (Tabs)
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
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนอุบัติเหตุทั้งหมด", f"{len(df):,} ครั้ง")
        if 'ระดับความเสี่ยง' in df.columns:
            c2.metric("เสี่ยงสูง (High Risk) 🔴", f"{len(df[df['ระดับความเสี่ยง']=='เสี่ยงสูง']):,} ครั้ง")
            c3.metric("เสี่ยงต่ำ (Low Risk) 🟢", f"{len(df[df['ระดับความเสี่ยง']=='เสี่ยงต่ำ']):,} ครั้ง")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**สัดส่วนความรุนแรง**")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='ระดับความเสี่ยง', palette=['#28B463', '#D62728'], ax=ax)
            st.pyplot(fig)
        with col_g2:
            st.write("**จำนวนอุบัติเหตุแยกตามช่วงเวลา (ไล่สี)**")
            if 'ช่วงเวลา' in df.columns:
                counts = df['ช่วงเวลา'].value_counts()
                fig2, ax2 = plt.subplots()
                pal = sns.color_palette("Blues_r", len(counts))
                sns.barplot(y=counts.index, x=counts.values, palette=pal, ax=ax2)
                st.pyplot(fig2)
    else:
        st.error("⚠️ ไม่พบไฟล์ข้อมูล 'Data_2Class_V1.csv' กรุณาตรวจสอบในโฟลเดอร์หรือ GitHub")

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
        st.error("🚨 ไม่พบไฟล์โมเดล AI (.pkl) กรุณาตรวจสอบการอัปโหลด")
    else:
        col_in, col_res = st.columns([1, 1])
        with col_in:
            st.write("### 📝 ระบุรายละเอียดอุบัติเหตุ")
            with st.form("ml_form"):
                time_val = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                weather_val = st.selectbox("สภาพอากาศ", ["แจ่มใส", "ฝนตก", "หมอกทึบ", "ไม่ระบุ"])
                mc_val = st.number_input("รถจักรยานยนต์ (คัน)", 0, 10, 1)
                car_val = st.number_input("รถยนต์นั่งส่วนบุคคล (คัน)", 0, 10, 0)
                submit_pred = st.form_submit_button("วิเคราะห์ความรุนแรง 🔍")
        
        with col_res:
            st.write("### 📊 ผลการทำนาย")
            if submit_pred:
                st.success("โมเดลกำลังประมวลผล... (ผลลัพธ์จะแสดงที่นี่)")
                # โค้ดส่วนการ Predict จริงจะอยู่ตรงนี้ (ใช้ model.predict)

# ------------------------------------------
# TAB 4: จัดการข้อมูล (CRUD - รูปแบบเก่า)
# ------------------------------------------
with tab4:
    if not st.session_state['logged_in']:
        st.warning("🔒 กรุณาเข้าสู่ระบบเพื่อเข้าถึงฐานข้อมูล")
    else:
        st.write("### 🗃️ ฐานข้อมูลอุบัติเหตุ (CRUD Management)")
        
        if df is not None:
            # 1. Read: ค้นหาและแสดงผล
            search = st.text_input("🔍 ค้นหาข้อมูล (จังหวัด, ช่วงเวลา, ฯลฯ)")
            if search:
                filtered_df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"แสดงข้อมูล 100 รายการล่าสุด จากทั้งหมด {len(df):,} รายการ")
            
            st.markdown("---")
            
            # 2. Create & Update/Delete
            col_c, col_ud = st.columns(2)
            
            with col_c:
                st.write("#### ➕ เพิ่มข้อมูลใหม่")
                with st.form("add_form"):
                    new_prov = st.text_input("จังหวัด")
                    new_time = st.selectbox("ช่วงเวลาเกิดเหตุ", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                    new_lat = st.number_input("ละติจูด", format="%.6f")
                    new_lon = st.number_input("ลองจิจูด", format="%.6f")
                    if st.form_submit_button("บันทึกข้อมูล"):
                        st.toast("บันทึกข้อมูลสำเร็จ! (Demo Mode)")
            
            with col_ud:
                st.write("#### ✏️ แก้ไข หรือ ลบข้อมูล")
                idx_to_edit = st.number_input("ระบุลำดับ (Index) ที่ต้องการจัดการ", 0, len(df)-1, 0)
                st.write("**ข้อมูลที่เลือก:**", df.iloc[idx_to_edit][['จังหวัด', 'ช่วงเวลา']].to_dict())
                
                c_edit, c_del = st.columns(2)
                with c_edit:
                    if st.button("🔄 อัปเดตข้อมูล", use_container_width=True):
                        st.info(f"อัปเดตข้อมูลลำดับที่ {idx_to_edit} แล้ว")
                with c_del:
                    if st.button("🗑️ ลบข้อมูลนี้", use_container_width=True, type="primary"):
                        st.error(f"ลบข้อมูลลำดับที่ {idx_to_edit} แล้ว")
        else:
            st.error("ไม่สามารถจัดการข้อมูลได้เนื่องจากไม่พบไฟล์ CSV")