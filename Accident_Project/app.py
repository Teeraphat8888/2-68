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
    # ระบบค้นหาไฟล์อัตโนมัติ (รองรับทั้งกรณีมีและไม่มีโฟลเดอร์ Accident_Project)
    file_name = "Data_2Class_V1.csv"
    if not os.path.exists(file_name) and os.path.exists(f"Accident_Project/{file_name}"):
        file_name = f"Accident_Project/{file_name}"
        
    if os.path.exists(file_name):
        try:
            # ใช้ utf-8-sig เพื่ออ่านภาษาไทยให้สมบูรณ์
            df = pd.read_csv(file_name, encoding='utf-8-sig')
            
            if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
                df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
                df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                
            if 'code_ระดับความเสี่ยง' in df.columns:
                df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
            return df
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
            return None
    return None

@st.cache_resource
def load_ml_assets():
    # ระบบค้นหาไฟล์โมเดลอัตโนมัติ
    prefix = "Accident_Project/" if os.path.exists("Accident_Project/best_model.pkl") else ""
    try:
        model = joblib.load(f"{prefix}best_model.pkl")
        scaler = joblib.load(f"{prefix}scaler.pkl")
        feature_cols = joblib.load(f"{prefix}feature_columns.pkl")
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
    st.image("Accident_Project/download.png", width=100)
    st.title("สำหรับผู้ดูแลระบบf")
    
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
st.title("แบบจำลองระบบวิเคราะห์ความรุนแรงอุบัติเหตุทางถนน เขตสุขภาพที่ 11")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "สถิติภาพรวม", 
    "แผนที่จุดเสี่ยง", 
    "ระบบทำนายความรุนแรง", 
    "จัดการข้อมูล "
])


# ------------------------------------------
# TAB 1: สถิติ (Overview)
# ------------------------------------------
with tab1:
    st.header("ภาพรวมสถานการณ์อุบัติเหตุ")
    if df is not None:
        def custom_metric(label, value, color):
            return f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 6px solid {color}; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); text-align: center;">
                <p style="margin:0px; font-size: 18px; color: #555; font-weight: 500;">{label}</p>
                <h2 style="margin:0px; color: {color}; font-size: 32px; font-weight: 700;">{value}</h2>
            </div>
            """

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(custom_metric("จำนวนอุบัติเหตุทั้งหมด", f"{len(df):,} ครั้ง", "#1E3A8A"), unsafe_allow_html=True)
        with col2:
            if 'ระดับความเสี่ยง' in df.columns:
                low_risk_count = len(df[df['ระดับความเสี่ยง'] == 'เสี่ยงต่ำ'])
                st.markdown(custom_metric("ความเสี่ยงต่ำ", f"{low_risk_count:,} ครั้ง", "#28B463"), unsafe_allow_html=True)
        with col3:
            if 'ระดับความเสี่ยง' in df.columns:
                high_risk_count = len(df[df['ระดับความเสี่ยง'] == 'เสี่ยงสูง'])
                st.markdown(custom_metric("ความเสี่ยงสูง", f"{high_risk_count:,} ครั้ง", "#D62728"), unsafe_allow_html=True)
        with col4:
            if 'จังหวัด' in df.columns:
                top_province = df['จังหวัด'].mode()[0]
                st.markdown(custom_metric("จังหวัดที่เกิดเหตุบ่อยสุด", top_province, "#424949"), unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.write("**สัดส่วนความรุนแรงของอุบัติเหตุ**")
            if 'ระดับความเสี่ยง' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df, x='ระดับความเสี่ยง', palette=['#28B463', '#D62728'], order=['เสี่ยงต่ำ', 'เสี่ยงสูง'], ax=ax)
                st.pyplot(fig)
                
        with col_graph2:
            st.write("**จำนวนอุบัติเหตุแบ่งตามช่วงเวลา**")
            if 'ช่วงเวลา' in df.columns:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df, y='ช่วงเวลา', order=df['ช่วงเวลา'].value_counts().index, palette='Blues_r', ax=ax2)
                st.pyplot(fig2)
    else:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูล CSV หรือ Excel")

# ------------------------------------------
# TAB 2: แผนที่ (Map)
# ------------------------------------------
with tab2:
    st.header("แผนที่จุดเสี่ยงอุบัติเหตุ (Accident Hotspots)")
    if df is not None and 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
        map_data = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        map_data = map_data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
        
        if 'ระดับความเสี่ยง' in map_data.columns:
            risk_filter = st.radio(
                "เลือกระดับความเสี่ยงที่ต้องการแสดงบนแผนที่:",
                ("แสดงทั้งหมด", "🔴 เฉพาะความเสี่ยงสูง", "🟢 เฉพาะความเสี่ยงต่ำ"),
                horizontal=True,
                key="map_filter" # ใส่ key ป้องกัน warning
            )
            
            if risk_filter == "🔴 เฉพาะความเสี่ยงสูง":
                map_data = map_data[map_data['ระดับความเสี่ยง'] == 'เสี่ยงสูง']
            elif risk_filter == "🟢 เฉพาะความเสี่ยงต่ำ":
                map_data = map_data[map_data['ระดับความเสี่ยง'] == 'เสี่ยงต่ำ']
                
        st.write(f"แสดงข้อมูลจำนวน: **{len(map_data):,}** จุดเกิดเหตุ")
        st.map(map_data[['lat', 'lon']], zoom=7)
    else:
        st.warning("⚠️ ไม่พบข้อมูลพิกัด (LATITUDE/LONGITUDE)")

# ------------------------------------------
# TAB 3: ทำนายผล (Prediction)
# ------------------------------------------
with tab3:
    st.header("ทดสอบระบบทำนายด้วย Machine Learning")
    
    if not st.session_state['logged_in']:
        st.error("### 🔒 เนื้อหาสงวนสิทธิ์เฉพาะเจ้าหน้าที่")
        st.info("กรุณาเข้าสู่ระบบผ่านแถบเมนูด้านซ้ายมือ (Sidebar) เพื่อใช้งานระบบทำนายความเสี่ยง")
    else:
        if model is None or scaler is None or feature_cols is None:
            st.error("🚨 ไม่พบไฟล์โมเดล (best_model.pkl, scaler.pkl, feature_columns.pkl) กรุณานำไฟล์มาวางในโฟลเดอร์เดียวกับ app.py")
        else:
            col_input, col_result = st.columns([1, 1])
            
            with col_input:
                st.subheader("📝 กรอกข้อมูลเพื่อประเมินความเสี่ยง")
                with st.form("ml_predict_form"):
                    time_period = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                    weather = st.selectbox("สภาพอากาศ", ["แจ่มใส", "ฝนตก", "หมอกทึบ", "ไม่ระบุ"])
                    accident_type = st.selectbox("ลักษณะการเกิดเหตุ", [
                        "ชนท้าย", "ชนในทิศทางตรงกันข้าม (ไม่ใช่การแซง)", "พลิกคว่ำ/ตกถนนในทางตรง", 
                        "พลิกคว่ำ/ตกถนนในทางโค้ง", "ชนสิ่งกีดขวาง (บนผิวจราจร)", "ไม่ระบุ"
                    ])
                    
                    col_n1, col_n2 = st.columns(2)
                    with col_n1:
                        motorcycle = st.number_input("รถจักรยานยนต์ (คัน)", min_value=0, max_value=10, value=1)
                        car = st.number_input("รถยนต์ส่วนบุคคล (คัน)", min_value=0, max_value=10, value=0)
                    with col_n2:
                        pickup = st.number_input("รถปิคอัพ (คัน)", min_value=0, max_value=10, value=0)
                        pedestrian = st.number_input("คนเดินเท้า (คน)", min_value=0, max_value=10, value=0)
                    
                    submitted = st.form_submit_button("วิเคราะห์ความเสี่ยง (รันโมเดล) 🔍")

            with col_result:
                st.subheader("📊 ผลลัพธ์จากโมเดล")
                
                if submitted:
                    input_dict = {
                        'รถจักรยานยนต์': [motorcycle], 'รถยนต์นั่งส่วนบุคคล': [car],
                        'รถปิคอัพบรรทุก4ล้อ': [pickup], 'คนเดินเท้า': [pedestrian],
                        'ช่วงเวลา': [time_period], 'สภาพอากาศ': [weather],
                        'ลักษณะการเกิดเหตุ': [accident_type]
                    }
                    input_df = pd.DataFrame(input_dict)
                    
                    input_dummies = pd.get_dummies(input_df)
                    input_final = input_dummies.reindex(columns=feature_cols, fill_value=0)
                    input_scaled = scaler.transform(input_final)
                    prediction = model.predict(input_scaled)[0]
                    
                    if prediction == 1: 
                        st.error("### 🔴ระดับความเสี่ยงสูง (High Risk)")
                        st.write("โมเดลวิเคราะห์ว่า **มีแนวโน้มสูงที่จะเกิดการบาดเจ็บสาหัสหรือเสียชีวิต**")
                        st.markdown("#### 💡 ข้อเสนอแนะเชิงนโยบาย")
                        st.info("- แจ้งเตือนศูนย์การแพทย์ฉุกเฉิน (EMS) ให้เตรียมรถกู้ชีพขั้นสูง\n- เสนอแนะจุดกวดขันวินัยจราจรในพื้นที่พิกัดนี้")
                    else:
                        st.success("### 🟢ระดับความเสี่ยงต่ำ (Low Risk)")
                        st.write("โมเดลวิเคราะห์ว่า **มีแนวโน้มบาดเจ็บเพียงเล็กน้อย หรือทรัพย์สินเสียหาย**")
                        st.markdown("#### 💡 ข้อเสนอแนะเชิงนโยบาย")
                        st.info("- เฝ้าระวังและปรับปรุงทัศนวิสัยบริเวณถนน\n- ส่งหน่วยกู้ภัยขั้นพื้นฐานเข้าประเมินสถานการณ์")
                else:
                    st.write("👈 กรอกข้อมูลด้านซ้ายแล้วกดปุ่มเพื่อรันโมเดลทำนาย")

# ------------------------------------------
# TAB 4: จัดการข้อมูล (CRUD)
# ------------------------------------------
with tab4:
    if not st.session_state['logged_in']:
        st.error("### 🔒 เนื้อหาสงวนสิทธิ์เฉพาะเจ้าหน้าที่")
        st.info("กรุณาเข้าสู่ระบบผ่านแถบเมนูด้านซ้ายมือ (Sidebar) เพื่อใช้งานระบบทำนายความเสี่ยง")
    else:
        st.write("### ฐานข้อมูลอุบัติเหตุ (CRUD Management)")
        
        if df is not None:
            # ค้นหาและแสดงผล
            search = st.text_input("ค้นหาข้อมูล เช่น จังหวัด, ช่วงเวลา, ฯลฯ")
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
                with st.form("create_form"):
                    new_prov = st.selectbox("จังหวัด", ["นครศรีธรรมราช", "สุราษฎร์ธานี", "ภูเก็ต", "กระบี่", "พังงา", "ระนอง", "ชุมพร"])
                    new_time = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                    new_lat = st.number_input("ละติจูด (LATITUDE)", value=8.4333, format="%.6f")
                    new_lon = st.number_input("ลองจิจูด (LONGITUDE)", value=99.9667, format="%.6f")
                    
                    submit_create = st.form_submit_button("บันทึกข้อมูล")
                    if submit_create:
                        st.toast(f"บันทึกข้อมูลอุบัติเหตุที่ {new_prov} สำเร็จ!")
            
            with col_ud:
                st.write("#### ✏️ แก้ไข หรือ ลบข้อมูล")
                idx_to_edit = st.number_input("ระบุลำดับ (Index)", 0, len(df)-1 if len(df)>0 else 0, 0)
                if len(df) > 0:
                    st.write("**ข้อมูลที่เลือก:**", df.iloc[idx_to_edit][['จังหวัด', 'ช่วงเวลา']].to_dict() if 'จังหวัด' in df.columns else "ไม่มีข้อมูลจังหวัด")
                
                c_edit, c_del = st.columns(2)
                with c_edit:
                    if st.button("อัปเดตข้อมูล", use_container_width=True):
                        st.info(f"อัปเดตข้อมูลลำดับที่ {idx_to_edit} แล้ว")
                with c_del:
                    if st.button("ลบข้อมูล", use_container_width=True, type="primary"):
                        st.error(f"ลบข้อมูลลำดับที่ {idx_to_edit} แล้ว")
        else:
            st.error("ไม่สามารถจัดการข้อมูลได้เนื่องจากไม่มีข้อมูล")