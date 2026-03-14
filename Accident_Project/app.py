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
# 1. ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ
# ==========================================
font_path = "Sarabun-Regular.ttf"
if not os.path.exists(font_path):
    url = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    urllib.request.urlretrieve(url, font_path)

fm.fontManager.addfont(font_path)
mpl.rc('font', family='Sarabun')
mpl.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 2. ตั้งค่าหน้าเว็บ
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_filenames = ['Data_2Class_V1.csv', 'Data_2Class_V1.csv.csv', 'Data_2Class_V1']
    
    for filename in possible_filenames:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
                    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
                    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
                if 'code_ระดับความเสี่ยง' in df.columns:
                    df['ระดับความเสี่ยง'] = df['code_ระดับความเสี่ยง'].map({1: 'เสี่ยงต่ำ', 2: 'เสี่ยงสูง'})
                return df
            except UnicodeDecodeError:
                try:
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(current_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        return None, None

df = load_data()
model, scaler = load_ml_assets()

# ฟังก์ชันดึงค่า Unique ตัวเลือก
def get_options(col_name, default_list):
    if df is not None and col_name in df.columns:
        return sorted([str(x) for x in df[col_name].dropna().unique()])
    return default_list

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
                        if user == "admin" and pw == "admin1111":
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
    "🗺️ แผนที่จุดเกิดเหตุ", 
    "🚨 ระบบทำนายความรุนแรง", 
    "📝 จัดการข้อมูล (CRUD)"
])

# ------------------------------------------
# TAB 1: สถิติ (Overview)
# ------------------------------------------
with tab1:
    if df is not None:
        st.markdown("### 📊 ภาพรวมสถิติอุบัติเหตุทางถนน")
        col1, col2, col3, col4 = st.columns(4)
        total_acc = len(df)
        high_risk = len(df[df['ระดับความเสี่ยง'] == 'เสี่ยงสูง']) if 'ระดับความเสี่ยง' in df.columns else 0
        low_risk = len(df[df['ระดับความเสี่ยง'] == 'เสี่ยงต่ำ']) if 'ระดับความเสี่ยง' in df.columns else 0
        total_dead = int(df['ผู้เสียชีวิต'].sum()) if 'ผู้เสียชีวิต' in df.columns else 0
        
        col1.metric("🚨 จำนวนอุบัติเหตุรวม", f"{total_acc:,} ครั้ง")
        col2.metric("🔴 รุนแรงสูง (High Severity)", f"{high_risk:,} ครั้ง")
        col3.metric("🟢 รุนแรงต่ำ (Low Severity)", f"{low_risk:,} ครั้ง")
        col4.metric("💀 ผู้เสียชีวิตรวม", f"{total_dead:,} คน")
        st.markdown("---")
        
        st.markdown("#### 📈 วิเคราะห์ปัจจัยการเกิดอุบัติเหตุ")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**สัดส่วนระดับความรุนแรง**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            if 'ระดับความเสี่ยง' in df.columns:
                sns.countplot(data=df, x='ระดับความเสี่ยง', palette=['#FF2B2B', '#09AB3B'], ax=ax1, order=['เสี่ยงสูง', 'เสี่ยงต่ำ'])
                ax1.set_ylabel("จำนวน (ครั้ง)")
                ax1.set_xlabel("")
                ax1.set_xticklabels(['รุนแรงสูง', 'รุนแรงต่ำ'])
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
    st.header("🤖 ระบบพยากรณ์ความรุนแรงของอุบัติเหตุด้วย AI")
    st.write("ระบุตัวแปรสภาพแวดล้อมและยานพาหนะให้ครบถ้วน เพื่อให้ AI ประเมินระดับความรุนแรงล่วงหน้า")

    if not st.session_state.get('logged_in', False):

        st.error("### 🔒 เนื้อหาสงวนสิทธิ์เฉพาะเจ้าหน้าที่")
        st.info("กรุณาเข้าสู่ระบบผ่านแถบเมนูด้านซ้ายมือ (Sidebar)")

    else:

        if model is None or scaler is None:

            st.error("🚨 ไม่พบไฟล์โมเดล AI")

        else:

            col_input, col_result = st.columns([1.2,1])

            # -----------------------------
            # INPUT
            # -----------------------------
            with col_input:

                st.subheader("📝 ระบุปัจจัยแวดล้อม")

                with st.form("ml_predict_form"):

                    c1,c2 = st.columns(2)

                    with c1:

                        time_period = st.selectbox(
                            "ช่วงเวลา",
                            get_options('ช่วงเวลา',
                            ["เช้า","สาย","บ่าย","เย็น","กลางคืน"])
                        )
                        weather = st.selectbox(
                            "สภาพอากาศ",
                            get_options('สภาพอากาศ',
                            ["แจ่มใส","ฝนตก","ไม่ระบุ"])
                        )

                    with c2:

                        location_type = st.selectbox(
                            "บริเวณที่เกิดเหตุ",
                            get_options('บริเวณที่เกิดเหตุ',
                            ["ทางตรง","ทางโค้ง","ทางแยก"])
                        )

                        presumed_cause = st.selectbox(
                            "มูลเหตุสันนิษฐาน",
                            get_options('มูลเหตุสันนิษฐาน',
                            ["ขับรถเร็วเกินกำหนด","เมาสุรา","ตัดหน้ากระชั้นชิด"])
                        )

                        accident_type = st.selectbox(
                            "ลักษณะการเกิดเหตุ",
                            get_options('ลักษณะการเกิดเหตุ',
                            ["ชนท้าย","พลิกคว่ำ","ชนสิ่งกีดขวาง"])
                        )

                    st.markdown("---")
                    st.subheader("🚗 ยานพาหนะที่เกี่ยวข้อง")

                    v1,v2,v3 = st.columns(3)

                    with v1:

                        v_moto = st.number_input("รถจักรยานยนต์",0,50,1)
                        v_car = st.number_input("รถยนต์นั่งส่วนบุคคล",0,50,0)
                        v_pick_pass = st.number_input("รถปิคอัพโดยสาร",0,50,0)
                        v_truck6 = st.number_input("รถบรรทุก6ล้อ",0,50,0)
                        v_etan = st.number_input("รถอีแต๋น",0,50,0)

                    with v2:

                        v_tri = st.number_input("รถสามล้อเครื่อง",0,50,0)
                        v_van = st.number_input("รถตู้",0,50,0)
                        v_bus = st.number_input("รถโดยสารมากกว่า4ล้อ",0,50,0)
                        v_truck10 = st.number_input("รถบรรทุกไม่เกิน10ล้อ",0,50,0)
                        v_other = st.number_input("รถอื่นๆ",0,50,0)

                    with v3:

                        v_pick_freight = st.number_input("รถปิคอัพบรรทุก4ล้อ",0,50,0)
                        v_truck_more10 = st.number_input("รถบรรทุกมากกว่า10ล้อ",0,50,0)
                        pedestrian = st.number_input("คนเดินเท้า",0,50,0)

                    submit_pred = st.form_submit_button(
                        "วิเคราะห์ความรุนแรงด้วย AI 🔍",
                        type="primary",
                        use_container_width=True
                    )

            # -----------------------------
            # RESULT
            # -----------------------------
            with col_result:

                st.subheader("🎯 ผลการทำนาย")

                if submit_pred:

                    with st.spinner("กำลังประมวลผล..."):

                        try:

                            # สร้าง dataframe
                            correct_features = scaler.feature_names_in_

                            input_final = pd.DataFrame(
                                0,
                                index=[0],
                                columns=correct_features
                            )

                            # -----------------------------
                            # ตัวเลข
                            # -----------------------------
                            numeric_inputs = {

                                'รถจักรยานยนต์':v_moto,
                                'รถสามล้อเครื่อง':v_tri,
                                'รถยนต์นั่งส่วนบุคคล':v_car,
                                'รถตู้':v_van,
                                'รถปิคอัพโดยสาร':v_pick_pass,
                                'รถโดยสารมากกว่า4ล้อ':v_bus,
                                'รถปิคอัพบรรทุก4ล้อ':v_pick_freight,
                                'รถบรรทุก6ล้อ':v_truck6,
                                'รถบรรทุกไม่เกิน10ล้อ':v_truck10,
                                'รถบรรทุกมากกว่า10ล้อ':v_truck_more10,
                                'รถอีแต๋น':v_etan,
                                'รถอื่นๆ':v_other,
                                'คนเดินเท้า':pedestrian
                            }

                            for col,val in numeric_inputs.items():

                                if col in input_final.columns:
                                    input_final[col] = val


                            # -----------------------------
                            # Categorical
                            # -----------------------------
                            cat_inputs = {

                                'ช่วงเวลา':time_period,
                                'บริเวณที่เกิดเหตุ':location_type,
                                'มูลเหตุสันนิษฐาน':presumed_cause,
                                'ลักษณะการเกิดเหตุ':accident_type,
                                'สภาพอากาศ':weather
                            }

                            for cat_col,cat_val in cat_inputs.items():

                                dummy_name = f"{cat_col}_{cat_val}"

                                if dummy_name in input_final.columns:
                                    input_final[dummy_name] = 1


                            # -----------------------------
                            # Scale
                            # -----------------------------
                            input_scaled = scaler.transform(input_final)


                            # -----------------------------
                            # Probability
                            # -----------------------------
                            proba = model.predict_proba(input_scaled)[0][1]

                            st.write(f"🔎 ความเสี่ยงความรุนแรง: {proba*100:.2f}%")


                            # -----------------------------
                            # Threshold
                            # -----------------------------
                            if proba >= 0.20:

                                st.error("### 🔴 ระดับความรุนแรง: สูง (High Severity)")
                                st.write("AI ประเมินว่าเหตุการณ์นี้มีแนวโน้มรุนแรงสูง")

                            else:

                                st.success("### 🟢 ระดับความรุนแรง: ต่ำ (Low Severity)")
                                st.write("AI ประเมินว่าเหตุการณ์นี้มีแนวโน้มรุนแรงต่ำ")


                        except Exception as e:

                            st.error("⚠️ เกิดข้อผิดพลาดในการทำนาย")
                            st.code(e)

                else:

                    st.info("👈 กรอกข้อมูลด้านซ้ายแล้วกดปุ่มวิเคราะห์")

# ------------------------------------------
# TAB 4: จัดการข้อมูล (CRUD)
# ------------------------------------------
with tab4:
    if not st.session_state['logged_in']:
        st.warning("🔒 กรุณาเข้าสู่ระบบเพื่อเข้าถึงฐานข้อมูล")
    else:
        st.write("### 🗃️ ฐานข้อมูลอุบัติเหตุ (CRUD Management)")
        
        if df is not None:
            search = st.text_input("🔍 ค้นหาข้อมูล (จังหวัด, ช่วงเวลา, ฯลฯ)")
            if search:
                filtered_df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"แสดงข้อมูล 100 รายการล่าสุด จากทั้งหมด {len(df):,} รายการ")
            
            st.markdown("---")
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