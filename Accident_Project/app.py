import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ==========================================
st.set_page_config(page_title="Road Safety Dashboard - Health Region 11", page_icon="🚑", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Sarabun', sans-serif !important; }
        h1, h2, h3 { font-weight: 700 !important; color: #1E3A8A !important; }
        .stButton>button { border-radius: 8px; }
        /* ปรับแต่งตาราง */
        .dataframe { font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. จัดการระบบ Login (Session State)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# สร้างแถบด้านข้าง (Sidebar) สำหรับ Login
with st.sidebar: 
    st.title("🔐 สำหรับเจ้าหน้าที่")
    
    if not st.session_state['logged_in']:
        st.write("กรุณาเข้าสู่ระบบเพื่อใช้งานระบบทำนายและจัดการข้อมูล")
        username = st.text_input("ชื่อผู้ใช้งาน (Username)")
        password = st.text_input("รหัสผ่าน (Password)", type="password")
        login_btn = st.button("เข้าสู่ระบบ", use_container_width=True)
        
        if login_btn:
            if username == "admin" and password == "admin123":
                st.session_state['logged_in'] = True
                st.success("เข้าสู่ระบบสำเร็จ!")
                st.rerun() 
            else:
                st.error("ชื่อผู้ใช้งานหรือรหัสผ่านไม่ถูกต้อง!")
    else:
        st.success("✅ เข้าสู่ระบบในฐานะ: เจ้าหน้าที่ (Admin)")
        logout_btn = st.button("ออกจากระบบ", use_container_width=True)
        if logout_btn:
            st.session_state['logged_in'] = False
            st.rerun()

# ==========================================
# 3. ฟังก์ชันโหลดไฟล์ทั้งหมด (Data + ML Assets)
# ==========================================
# เปลี่ยนจาก @st.cache_data เป็นแบบดึงใหม่เสมอ เพื่อให้เวลาอัปเดตข้อมูลแล้วหน้าเว็บเปลี่ยนตามทันที
def load_data():
    file_name = "Data_2Class_V1.xlsx" 
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LATITUDE'] = df['LATITUDE'].fillna(df['LATITUDE'].median())
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            df['LONGITUDE'] = df['LONGITUDE'].fillna(df['LONGITUDE'].median())
        if 'code_ระดับความเสี่ยง' in df.columns:
            df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
        return df
    return None

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, None

df = load_data()
model, scaler, feature_cols = load_ml_assets()

# ==========================================
# 4. ส่วนหัวของ Dashboard
# ==========================================
st.title("ระบบวิเคราะห์และทำนายความรุนแรงอุบัติเหตุทางถนน เขตสุขภาพที่ 11")
st.markdown("---")

# ==========================================
# 5. สร้างเมนู Tab แบ่งหน้าจอ (เพิ่ม Tab ที่ 4)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "ภาพรวมสถิติ (Overview)", 
    "แผนที่จุดเสี่ยง (Hotspots)", 
    "ระบบทำนาย (Prediction)", 
    "จัดการข้อมูล "  # แท็บใหม่
])

# ------------------------------------------
# TAB 1: ภาพรวมสถิติ (Overview)
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
        st.warning("⚠️ ไม่พบไฟล์ Excel ข้อมูล")

# ------------------------------------------
# TAB 2: แผนที่จุดเสี่ยง (Hotspots)
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
# TAB 3: ระบบทำนายด้วยโมเดลจริง (ML Prediction)
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
                        st.error("### 🔴 ผลการทำนาย: ระดับความเสี่ยงสูง (High Risk)")
                        st.write("โมเดลวิเคราะห์ว่า: **มีแนวโน้มสูงที่จะเกิดการบาดเจ็บสาหัสหรือเสียชีวิต**")
                        st.markdown("#### 💡 ข้อเสนอแนะเชิงนโยบาย")
                        st.info("- แจ้งเตือนศูนย์การแพทย์ฉุกเฉิน (EMS) ให้เตรียมรถกู้ชีพขั้นสูง\n- เสนอแนะจุดกวดขันวินัยจราจรในพื้นที่พิกัดนี้")
                    else:
                        st.success("### 🟢 ผลการทำนาย: ระดับความเสี่ยงต่ำ (Low Risk)")
                        st.write("โมเดลวิเคราะห์ว่า: **มีแนวโน้มบาดเจ็บเพียงเล็กน้อย หรือทรัพย์สินเสียหาย**")
                        st.markdown("#### 💡 ข้อเสนอแนะเชิงนโยบาย")
                        st.info("- เฝ้าระวังและปรับปรุงทัศนวิสัยบริเวณถนน\n- ส่งหน่วยกู้ภัยขั้นพื้นฐานเข้าประเมินสถานการณ์")
                else:
                    st.write("👈 กรอกข้อมูลด้านซ้ายแล้วกดปุ่มเพื่อรันโมเดลทำนาย")

# ------------------------------------------
# TAB 4: จัดการข้อมูล (CRUD Data Management)
# ------------------------------------------
with tab4:
    st.header("ระบบจัดการฐานข้อมูลอุบัติเหตุ ")
    
    # 🔒 ตรวจสอบสิทธิ์การเข้าใช้งาน
    if not st.session_state['logged_in']:
        st.error("### 🔒 เนื้อหาสงวนสิทธิ์เฉพาะเจ้าหน้าที่")
        st.info("กรุณาเข้าสู่ระบบผ่านแถบเมนูด้านซ้ายมือ (Sidebar) เพื่อจัดการข้อมูล")
    else:
        if df is not None:
            # --- ส่วนที่ 1: Read (แสดงข้อมูลทั้งหมด) ---
            st.subheader(" ฐานข้อมูลปัจจุบัน")
            st.write(f"จำนวนข้อมูลทั้งหมด: {len(df)} รายการ")
            
            # ช่องค้นหา (Search)
            search_query = st.text_input("🔍 ค้นหาข้อมูล (เช่น ชื่อจังหวัด, ช่วงเวลา)")
            if search_query:
                # ค้นหาแบบง่ายๆ ในคอลัมน์จังหวัดและช่วงเวลา
                filtered_df = df[df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)]
                st.dataframe(filtered_df, use_container_width=True, height=250)
            else:
                st.dataframe(df.head(100), use_container_width=True, height=250) # แสดงแค่ 100 แถวแรกเพื่อไม่ให้โหลดช้า
                st.caption("*แสดงผลข้อมูล 100 รายการล่าสุด")
            
            st.markdown("---")
            
            # --- ส่วนที่ 2: Create / Update / Delete ---
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                st.subheader("เพิ่มข้อมูลใหม่ (Create)")
                with st.form("create_form"):
                    new_prov = st.selectbox("จังหวัด", ["นครศรีธรรมราช", "สุราษฎร์ธานี", "ภูเก็ต", "กระบี่", "พังงา", "ระนอง", "ชุมพร"])
                    new_time = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                    new_lat = st.number_input("ละติจูด (LATITUDE)", value=8.4333, format="%.6f")
                    new_lon = st.number_input("ลองจิจูด (LONGITUDE)", value=99.9667, format="%.6f")
                    
                    create_btn = st.form_submit_button("บันทึกข้อมูลใหม่")
                    if create_btn:
                        # (โค้ดจำลองการเพิ่มข้อมูล - ในการใช้งานจริงต้องเขียนบันทึกลงไฟล์ Excel)
                        st.success("✅ บันทึกข้อมูลใหม่เรียบร้อยแล้ว! (หมายเหตุ: นี่คือการจำลองในหน้าจอ Demo)")
                        
            with col_action2:
                st.subheader("แก้ไข / ลบ ข้อมูล (Update & Delete)")
                # (โค้ดจำลองการเลือก ID เพื่อแก้ไขหรือลบ)
                row_id = st.number_input("ระบุลำดับ (Index) ที่ต้องการจัดการ", min_value=0, max_value=len(df)-1 if len(df)>0 else 0, value=0)
                
                st.write(f"**ตัวอย่างข้อมูลที่เลือก (Index {row_id}):**")
                if len(df) > 0:
                    st.write(df.iloc[row_id][['จังหวัด', 'ช่วงเวลา', 'LATITUDE', 'LONGITUDE']].to_dict())
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    update_btn = st.button("อัปเดตข้อมูลนี้", use_container_width=True)
                    if update_btn:
                        st.info(f"อัปเดตข้อมูลแถวที่ {row_id} สำเร็จ!")
                with col_btn2:
                    delete_btn = st.button("ลบข้อมูลนี้", use_container_width=True, type="primary")
                    if delete_btn:
                        st.warning(f"ลบข้อมูลแถวที่ {row_id} สำเร็จ!")

        else:
            st.warning("ไม่พบไฟล์ Excel ข้อมูล ไม่สามารถจัดการข้อมูลได้11")
            st.info(f"🔍 ลิสต์ไฟล์ทั้งหมดที่ระบบมองเห็นตอนนี้: {os.listdir()}")