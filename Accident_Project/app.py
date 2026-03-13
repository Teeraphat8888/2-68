import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
from sklearn.cluster import DBSCAN

# ==========================================
# 1. ตั้งค่าหน้าเพจและฟอนต์ (Page Config & CSS)
# ==========================================
st.set_page_config(page_title="ระบบวิเคราะห์และพยากรณ์อุบัติเหตุ", page_icon="🚦", layout="wide")

# ฝัง CSS เพื่อใช้ฟอนต์ Sarabun (ภาษาไทย)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Sarabun', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ฟังก์ชันโหลดข้อมูลและโมเดล AI (Caching)
# ==========================================
@st.cache_data
def load_data():
    try:
        # เปลี่ยนชื่อไฟล์ให้ตรงกับข้อมูลของคุณ
        df = pd.read_csv('Data_2Class_V1.csv')
        return df
    except Exception as e:
        return None

@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, None

df = load_data()
model, scaler, feature_cols = load_models()

# ==========================================
# 3. ระบบ Sidebar และ Login
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094865.png", width=100)
    st.title("เมนูจัดการระบบ")
    
    if not st.session_state['logged_in']:
        st.subheader("🔒 เข้าสู่ระบบ (สำหรับเจ้าหน้าที่)")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # กำหนดรหัสผ่านแบบง่าย (สามารถเปลี่ยนได้)
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
# 4. หน้าจอหลักและการแบ่ง Tabs
# ==========================================
st.title("🚦 ระบบวิเคราะห์และพยากรณ์ความเสี่ยงอุบัติเหตุทางถนน")

# สร้าง Tabs 3 หน้า
tab1, tab2, tab3 = st.tabs(["📊 ภาพรวมข้อมูล", "🗺️ แผนที่จุดเสี่ยง", "🤖 พยากรณ์ด้วย AI"])

# ------------------------------------------
# TAB 1: ภาพรวมข้อมูล (Dashboard)
# ------------------------------------------
with tab1:
    st.header("📊 แดชบอร์ดสรุปสถิติอุบัติเหตุทางถนน")
    st.write("ภาพรวมข้อมูลสถิติอุบัติเหตุทั้งหมดที่นำมาใช้ในการวิเคราะห์และสร้างโมเดล AI")
    
    if df is not None:
        # --- ส่วนที่ 1: กล่องตัวเลขสรุป (Key Metrics) ---
        st.markdown("### 📌 ข้อมูลสรุปความสูญเสีย")
        
        # คำนวณผลรวม (เช็คก่อนว่ามีคอลัมน์เหล่านี้ในข้อมูลไหม)
        total_acc = len(df)
        total_dead = int(df['ผู้เสียชีวิต'].sum()) if 'ผู้เสียชีวิต' in df.columns else 0
        total_severe = int(df['ผู้บาดเจ็บสาหัส'].sum()) if 'ผู้บาดเจ็บสาหัส' in df.columns else 0
        total_minor = int(df['ผู้บาดเจ็บเล็กน้อย'].sum()) if 'ผู้บาดเจ็บเล็กน้อย' in df.columns else 0
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("🚨 จำนวนอุบัติเหตุรวม", f"{total_acc:,} ครั้ง")
        col_kpi2.metric("💀 ผู้เสียชีวิตรวม", f"{total_dead:,} คน")
        col_kpi3.metric("🚑 ผู้บาดเจ็บสาหัสรวม", f"{total_severe:,} คน")
        col_kpi4.metric("🩹 ผู้บาดเจ็บเล็กน้อยรวม", f"{total_minor:,} คน")
        
        st.markdown("---")
        
        # --- ส่วนที่ 2: กราฟสถิติ (Charts) ---
        st.markdown("### 📈 กราฟวิเคราะห์ปัจจัยการเกิดอุบัติเหตุ")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**⏰ ช่วงเวลาที่เกิดอุบัติเหตุ**")
            if 'ช่วงเวลา' in df.columns:
                # นับจำนวนตามช่วงเวลาและสร้างกราฟแท่งสีส้ม
                time_counts = df['ช่วงเวลา'].value_counts()
                st.bar_chart(time_counts, color="#FF5733")
            else:
                st.info("ไม่พบข้อมูลคอลัมน์ 'ช่วงเวลา'")
                
        with col_chart2:
            st.markdown("**☁️ สภาพอากาศขณะเกิดเหตุ**")
            if 'สภาพอากาศ' in df.columns:
                # นับจำนวนตามสภาพอากาศและสร้างกราฟแท่งสีฟ้า
                weather_counts = df['สภาพอากาศ'].value_counts()
                st.bar_chart(weather_counts, color="#3399FF")
            else:
                st.info("ไม่พบข้อมูลคอลัมน์ 'สภาพอากาศ'")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- ส่วนที่ 3: ตารางข้อมูลดิบแบบพับเก็บได้ ---
        st.markdown("### 📋 ตารางข้อมูลดิบ (Raw Data)")
        with st.expander("คลิกเพื่อกางตารางข้อมูลที่ใช้ในการวิเคราะห์ 🔽"):
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning("⚠️ ไม่พบไฟล์ข้อมูล Data_2Class_V1.csv กรุณาอัปโหลดไฟล์เข้าสู่ระบบ")

# ------------------------------------------
# TAB 2: แผนที่จุดเสี่ยง (PyDeck Map)
# ------------------------------------------
with tab2:
    st.header("🗺️ แผนที่วิเคราะห์จุดเสี่ยงอุบัติเหตุ")
    
    if df is not None and 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
        map_data = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        map_data = map_data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
        
        try:
            # คำนวณระยะทางและจัดกลุ่มด้วย DBSCAN
            coords = np.radians(map_data[['lat', 'lon']].values)
            kms_per_radian = 6371.0088
            epsilon = 0.5 / kms_per_radian  # รัศมี 500 เมตร
            
            db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords)
            map_data['cluster'] = db.labels_
            
            # ยุบรวมจุด (Centroid) และนับจำนวน
            cluster_stats = map_data.groupby('cluster').agg(
                lat=('lat', 'mean'),      
                lon=('lon', 'mean'),      
                acc_count=('cluster', 'count') 
            ).reset_index()
            
            cluster_stats = cluster_stats.rename(columns={'acc_count': 'จำนวนอุบัติเหตุ'})
            cluster_stats['ระดับความเสี่ยง'] = np.where(cluster_stats['จำนวนอุบัติเหตุ'] >= 5, 'เสี่ยงสูง', 'เสี่ยงต่ำ')
            
            # กำหนดสีขอบและสีพื้นสำหรับวงกลมบนแผนที่
            cluster_stats['fill_color'] = cluster_stats['ระดับความเสี่ยง'].apply(
                lambda x: [255, 43, 43, 80] if x == 'เสี่ยงสูง' else [9, 171, 59, 80]
            )
            cluster_stats['line_color'] = cluster_stats['ระดับความเสี่ยง'].apply(
                lambda x: [255, 43, 43, 255] if x == 'เสี่ยงสูง' else [9, 171, 59, 255]
            )
            
            # คำนวณตัวเลข Metrics
            high_risk_zones = len(cluster_stats[cluster_stats['ระดับความเสี่ยง'] == 'เสี่ยงสูง'])
            low_risk_zones = len(cluster_stats[cluster_stats['ระดับความเสี่ยง'] == 'เสี่ยงต่ำ'])
            total_accidents_mapped = cluster_stats['จำนวนอุบัติเหตุ'].sum()
            
            # แสดง Metrics
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            col_sum1.metric("🔴 จุดเสี่ยงสูง (High Risk Zones)", f"{high_risk_zones:,} โซน")
            col_sum2.metric("🟢 จุดเสี่ยงต่ำ (Low Risk Zones)", f"{low_risk_zones:,} โซน")
            col_sum3.metric("📊 ครอบคลุมจำนวนอุบัติเหตุ", f"{total_accidents_mapped:,} ครั้ง")
            st.markdown("---")
            
            # ตัวกรองระดับความเสี่ยง
            filter_opt = st.radio(
                "เลือกระดับความเสี่ยงที่ต้องการแสดงบนแผนที่:",
                ("🌎 แสดงทั้งหมด", "🔴 เฉพาะจุดเสี่ยงสูง (≥ 5 ครั้ง)", "🟢 เฉพาะจุดเสี่ยงต่ำ (< 5 ครั้ง)"),
                horizontal=True
            )
            
            if filter_opt == "🔴 เฉพาะจุดเสี่ยงสูง (≥ 5 ครั้ง)":
                plot_data = cluster_stats[cluster_stats['ระดับความเสี่ยง'] == 'เสี่ยงสูง']
            elif filter_opt == "🟢 เฉพาะจุดเสี่ยงต่ำ (< 5 ครั้ง)":
                plot_data = cluster_stats[cluster_stats['ระดับความเสี่ยง'] == 'เสี่ยงต่ำ']
            else:
                plot_data = cluster_stats
                
            st.write(f"กำลังแสดงจุดศูนย์กลางบนแผนที่: **{len(plot_data):,}** โซน")
            
            # สร้าง PyDeck Layer
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=plot_data,
                get_position='[lon, lat]',
                get_radius=500,               
                get_fill_color='fill_color',  
                get_line_color='line_color',  
                stroked=True,                 
                filled=True,                  
                line_width_min_pixels=3,      
                pickable=True                 
            )

            view_state = pdk.ViewState(
                latitude=plot_data['lat'].mean() if len(plot_data) > 0 else 8.4333,
                longitude=plot_data['lon'].mean() if len(plot_data) > 0 else 99.9667,
                zoom=7, pitch=0
            )

            tooltip = {
                "html": "<b>ระดับความเสี่ยง:</b> {ระดับความเสี่ยง} <br/> <b>จำนวนอุบัติเหตุ:</b> <span style='color: yellow;'>{จำนวนอุบัติเหตุ}</span> ครั้ง",
                "style": {"backgroundColor": "#2C3E50", "color": "white", "font-family": "Sarabun, sans-serif", "border-radius": "8px", "padding": "10px"}
            }

            st.pydeck_chart(pdk.Deck(
                map_style='light',
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip
            ))
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้างแผนที่: {e}")
            st.map(map_data[['lat', 'lon']])
    else:
        st.warning("⚠️ ไม่พบข้อมูลพิกัด (LATITUDE/LONGITUDE) ในไฟล์ข้อมูล")

# ------------------------------------------
# TAB 3: ทำนายผล (Prediction)
# ------------------------------------------
with tab3:
    st.header("🤖 ระบบพยากรณ์ความรุนแรงของอุบัติเหตุด้วย AI")
    st.write("ระบบจะนำข้อมูลที่คุณกรอกไปประมวลผลผ่านโมเดล Machine Learning (`best_model.pkl`) เพื่อทำนายระดับความรุนแรง")
    
    if not st.session_state.get('logged_in', False):
        st.error("### 🔒 เนื้อหาสงวนสิทธิ์เฉพาะเจ้าหน้าที่")
        st.info("กรุณาเข้าสู่ระบบผ่านแถบเมนูด้านซ้ายมือ (Sidebar) เพื่อใช้งานระบบพยากรณ์ความเสี่ยง")
    else:
        if model is None or scaler is None or feature_cols is None:
            st.error("🚨 ไม่พบไฟล์โมเดล AI กรุณาตรวจสอบว่ามีไฟล์ `best_model.pkl`, `scaler.pkl`, และ `feature_columns.pkl` อยู่ในระบบ")
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
                    
                    st.markdown("**ยานพาหนะที่เกี่ยวข้อง**")
                    col_n1, col_n2 = st.columns(2)
                    with col_n1:
                        motorcycle = st.number_input("รถจักรยานยนต์ (คัน)", min_value=0, max_value=10, value=1)
                        car = st.number_input("รถยนต์นั่งส่วนบุคคล (คัน)", min_value=0, max_value=10, value=0)
                    with col_n2:
                        pickup = st.number_input("รถปิคอัพบรรทุก4ล้อ (คัน)", min_value=0, max_value=10, value=0)
                        pedestrian = st.number_input("คนเดินเท้า (คน)", min_value=0, max_value=10, value=0)
                    
                    st.markdown("**ข้อมูลผู้บาดเจ็บและเสียชีวิต**")
                    col_inj1, col_inj2, col_inj3 = st.columns(3)
                    with col_inj1:
                        minor_inj = st.number_input("ผู้บาดเจ็บเล็กน้อย (คน)", min_value=0, max_value=50, value=0)
                    with col_inj2:
                        severe_inj = st.number_input("ผู้บาดเจ็บสาหัส (คน)", min_value=0, max_value=50, value=0)
                    with col_inj3:
                        fatalities = st.number_input("ผู้เสียชีวิต (คน)", min_value=0, max_value=50, value=0)
                    
                    submitted = st.form_submit_button("พยากรณ์ความรุนแรง 🔍")

            with col_result:
                st.subheader("🎯 ผลการพยากรณ์จากโมเดล")
                
                if submitted:
                    with st.spinner('กำลังประมวลผลผ่านโมเดล AI...'):
                        # เตรียมข้อมูลพื้นฐาน (ใส่ LAT/LON กลางเป็นค่าตั้งต้นเพื่อไม่ให้โมเดล Scale เพี้ยน)
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
                            # แปลงข้อมูลแบบ 1-Hot Encoding (pd.get_dummies)
                            input_dummies = pd.get_dummies(input_df)
                            
                            # จัดคอลัมน์ให้ตรงกับโมเดลตอนสอน (เติม 0 ในคอลัมน์ที่ขาดหายไป)
                            input_final = input_dummies.reindex(columns=feature_cols, fill_value=0)
                            
                            # ปรับสเกลข้อมูล
                            input_scaled = scaler.transform(input_final)
                            
                            # รันทำนายผล
                            prediction = model.predict(input_scaled)[0]
                            
                            if prediction == 1: 
                                st.error("🚨 **AI พยากรณ์ว่า: ระดับความเสี่ยงสูง (High Risk)**")
                                st.write("โมเดลวิเคราะห์จากรูปแบบข้อมูลแล้วพบว่า **มีแนวโน้มสูงที่จะเกิดความสูญเสียรุนแรง**")
                                st.markdown("---")
                                st.info("**💡 ข้อเสนอแนะเชิงปฏิบัติการ:**\n- แจ้งศูนย์การแพทย์ฉุกเฉิน (EMS) พื้นที่ให้เตรียมพร้อมขั้นสูงสุด\n- ส่งเจ้าหน้าที่ตำรวจตรวจสอบและจัดการจราจรจุดเกิดเหตุทันที")
                            else:
                                st.success("✅ **AI พยากรณ์ว่า: ระดับความเสี่ยงต่ำ (Low Risk)**")
                                st.write("โมเดลวิเคราะห์จากรูปแบบข้อมูลแล้วพบว่า **มีแนวโน้มบาดเจ็บเพียงเล็กน้อย หรือมีเพียงทรัพย์สินเสียหาย**")
                                st.markdown("---")
                                st.info("**💡 ข้อเสนอแนะเชิงปฏิบัติการ:**\n- ส่งหน่วยกู้ภัยขั้นพื้นฐานเข้าประเมินสถานการณ์และปฐมพยาบาล\n- เคลียร์พื้นที่ผิวจราจรเพื่อป้องกันอุบัติเหตุซ้ำซ้อน")

                        except Exception as e:
                            st.error(f"⚠️ เกิดข้อผิดพลาดในการคำนวณของโมเดล:")
                            st.code(f"Error Details: {e}")
                else:
                    st.info("👈 กรอกข้อมูลอุบัติเหตุทางด้านซ้ายให้ครบถ้วน แล้วกดปุ่ม **พยากรณ์ความรุนแรง**")