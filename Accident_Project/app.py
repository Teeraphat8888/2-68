import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
import os

# ==========================================
# 1. ตั้งค่าหน้าเพจ
# ==========================================
st.set_page_config(page_title="ระบบวิเคราะห์อุบัติเหตุ", page_icon="🚦", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Sarabun', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ฟังก์ชันโหลดข้อมูลและโมเดล
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_filenames = ['Data_2Class_V1.csv', 'Data_2Class_V1.csv.csv', 'Data_2Class_V1']
    
    for filename in possible_filenames:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                return df
            except:
                try:
                    df = pd.read_csv(file_path, encoding='windows-874')
                    return df
                except:
                    pass
    return None

@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

df = load_data()
model, scaler = load_models()

# ==========================================
# 3. ระบบ Sidebar Login
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

with st.sidebar:
    st.title("เมนูจัดการระบบ")
    if not st.session_state['logged_in']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state['logged_in'] = True
                st.success("เข้าสู่ระบบสำเร็จ!")
                st.rerun()
            else:
                st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    else:
        st.success("✅ เข้าสู่ระบบในฐานะ Admin")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

# ==========================================
# 4. หน้าจอหลัก (Tabs)
# ==========================================
st.title("🚦 ระบบวิเคราะห์และพยากรณ์ความเสี่ยงอุบัติเหตุทางถนน")
tab1, tab2, tab3 = st.tabs(["📊 ภาพรวมข้อมูล", "🗺️ แผนที่จุดเสี่ยง", "🤖 พยากรณ์ด้วย AI"])

# ------------------------------------------
# TAB 1: ภาพรวมข้อมูล 
# ------------------------------------------
with tab1:
    st.header("📊 ภาพรวมข้อมูลสถิติอุบัติเหตุ")
    if df is not None:
        st.markdown(f"**จำนวนข้อมูลอุบัติเหตุในระบบ:** {len(df):,} รายการ")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.error("⚠️ ไม่พบไฟล์ข้อมูล Data_2Class_V1.csv")

# ------------------------------------------
# TAB 2: แผนที่ (เวอร์ชันแรกสุดแบบดั้งเดิม)
# ------------------------------------------
with tab2:
    st.header("🗺️ แผนที่วิเคราะห์จุดเสี่ยง (Accident Hotspots)")
    st.write("🎯 **นิยามจุดเสี่ยง:** บริเวณที่มีอุบัติเหตุเกิดขึ้น **ตั้งแต่ 5 ครั้งขึ้นไป ในรัศมี 500 เมตร**")
    
    if df is not None and 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
        map_data = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        map_data = map_data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
        
        try:
            # ใช้ DBSCAN หาระยะทาง
            coords = np.radians(map_data[['lat', 'lon']].values)
            kms_per_radian = 6371.0088
            epsilon = 0.5 / kms_per_radian
            
            # รันหา Cluster (>= 5 จุด ในระยะ 500m)
            db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(coords)
            map_data['cluster'] = db.labels_
            
            # แบ่งกลุ่มข้อมูล (-1 คือจุดปกติ, ค่าอื่นคือจุดเสี่ยง)
            map_data['is_hotspot'] = map_data['cluster'] != -1
            map_data['color'] = map_data['is_hotspot'].map({True: '#FF0000', False: '#28B463'})
            
            # ตัวกรอง
            filter_opt = st.radio(
                "รูปแบบการแสดงบนแผนที่:",
                ("🔴 แสดงเฉพาะจุดเสี่ยง (Hotspots)", "🟢 แสดงจุดปกติ", "🌎 แสดงทั้งหมด"),
                horizontal=True
            )
            
            if filter_opt == "🔴 แสดงเฉพาะจุดเสี่ยง (Hotspots)":
                plot_data = map_data[map_data['is_hotspot'] == True]
            elif filter_opt == "🟢 แสดงจุดปกติ":
                plot_data = map_data[map_data['is_hotspot'] == False]
            else:
                plot_data = map_data
                
            # สรุปตัวเลข
            hotspot_count = len(map_data[map_data['is_hotspot'] == True])
            cluster_count = map_data['cluster'].nunique() - (1 if -1 in map_data['cluster'].values else 0)
            
            col_sum1, col_sum2 = st.columns(2)
            col_sum1.metric("จำนวนอุบัติเหตุในพื้นที่จุดเสี่ยง", f"{hotspot_count:,} ครั้ง")
            col_sum2.metric("จำนวนกลุ่มจุดเสี่ยง (Hotspot Zones)", f"{cluster_count:,} โซน")
            
            st.write(f"กำลังแสดงข้อมูล **{len(plot_data):,}** จุด บนแผนที่")
            
            # พล็อตลงแผนที่แบบธรรมดา
            st.map(plot_data, latitude='lat', longitude='lon', color='color', zoom=7)
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
            st.map(map_data[['lat', 'lon']]) 
    else:
        st.warning("⚠️ ไม่พบข้อมูลพิกัด (LATITUDE/LONGITUDE) ในไฟล์ข้อมูล")

# ------------------------------------------
# TAB 3: ทำนายผล 
# ------------------------------------------
with tab3:
    st.header("🤖 ระบบพยากรณ์ความรุนแรงของอุบัติเหตุด้วย AI")
    
    if not st.session_state.get('logged_in', False):
        st.error("### 🔒 เนื้อหาสงวนสิทธิ์เฉพาะเจ้าหน้าที่")
    else:
        if model is None or scaler is None:
            st.error("🚨 ไม่พบไฟล์โมเดล AI `best_model.pkl` และ `scaler.pkl`")
        else:
            col_input, col_result = st.columns([1, 1])
            with col_input:
                st.subheader("📝 กรอกข้อมูลอุบัติเหตุ")
                with st.form("ml_predict_form"):
                    time_period = st.selectbox("ช่วงเวลา", ["เช้า", "สาย", "บ่าย", "เย็น", "กลางคืน"])
                    weather = st.selectbox("สภาพอากาศ", ["แจ่มใส", "ฝนตก", "หมอกทึบ", "ไม่ระบุ"])
                    accident_type = st.selectbox("ลักษณะการเกิดเหตุ", [
                        "ชนท้าย", "ชนในทิศทางตรงกันข้าม (ไม่ใช่การแซง)", "พลิกคว่ำ/ตกถนนในทางตรง", 
                        "พลิกคว่ำ/ตกถนนในทางโค้ง", "ชนสิ่งกีดขวาง (บนผิวจราจร)", "ไม่ระบุ"
                    ])
                    motorcycle = st.number_input("รถจักรยานยนต์ (คัน)", 0, 10, 1)
                    car = st.number_input("รถยนต์ (คัน)", 0, 10, 0)
                    pickup = st.number_input("รถปิคอัพ (คัน)", 0, 10, 0)
                    pedestrian = st.number_input("คนเดินเท้า (คน)", 0, 10, 0)
                    minor_inj = st.number_input("บาดเจ็บเล็กน้อย (คน)", 0, 50, 0)
                    severe_inj = st.number_input("บาดเจ็บสาหัส (คน)", 0, 50, 0)
                    fatalities = st.number_input("เสียชีวิต (คน)", 0, 50, 0)
                    
                    submitted = st.form_submit_button("พยากรณ์ 🔍")

            with col_result:
                st.subheader("🎯 ผลการพยากรณ์")
                if submitted:
                    input_dict = {
                        'รถจักรยานยนต์': [motorcycle], 'รถยนต์นั่งส่วนบุคคล': [car],
                        'รถปิคอัพบรรทุก4ล้อ': [pickup], 'คนเดินเท้า': [pedestrian],
                        'ช่วงเวลา': [time_period], 'สภาพอากาศ': [weather],
                        'ลักษณะการเกิดเหตุ': [accident_type],
                        'ผู้บาดเจ็บเล็กน้อย': [minor_inj],
                        'ผู้บาดเจ็บสาหัส': [severe_inj],
                        'ผู้เสียชีวิต': [fatalities],
                        'LATITUDE': [8.4333], 'LONGITUDE': [99.9667] 
                    }
                    input_df = pd.DataFrame(input_dict)
                    try:
                        correct_features = scaler.feature_names_in_
                        input_dummies = pd.get_dummies(input_df)
                        input_final = input_dummies.reindex(columns=correct_features, fill_value=0)
                        input_scaled = scaler.transform(input_final)
                        prediction = model.predict(input_scaled)[0]
                        
                        if prediction == 1: 
                            st.error("🚨 **ความเสี่ยงสูง (High Risk)**")
                        else:
                            st.success("✅ **ความเสี่ยงต่ำ (Low Risk)**")
                    except Exception as e:
                        st.error(f"Error: {e}")