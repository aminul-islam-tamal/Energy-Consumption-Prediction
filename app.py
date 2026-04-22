import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ১. মডেল এবং কলাম লোড করা
@st.cache_resource
def load_model():
    model = joblib.load('lgbm_energy_model.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, cols

model, model_columns = load_model()

# ২. অ্যাপের শিরোনাম
st.title("⚡ Energy Prediction App")
st.write("আপনার বিল্ডিং এবং আবহাওয়ার তথ্য দিন, আর আমরা বিদ্যুতের সম্ভাব্য ব্যবহার (Meter Reading) প্রেডিক্ট করব।")

# ৩. ইনপুট প্যানেল (সাইডবার)
st.sidebar.header("🏢 Building Info")
primary_use = st.sidebar.selectbox("Primary Use", ['Education', 'Office', 'Public services', 'Retail', 'Other', 'Parking', 'Warehouse/storage'])
square_feet = st.sidebar.number_input("Square Feet", min_value=100, max_value=1000000, value=50000)
year_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
floor_count = st.sidebar.number_input("Floor Count", min_value=1, max_value=50, value=1)

st.sidebar.header("🌤️ Weather Info")
air_temperature = st.sidebar.number_input("Air Temperature (°C)", value=25.0)
cloud_coverage = st.sidebar.slider("Cloud Coverage", min_value=0.0, max_value=10.0, value=6.0)

st.sidebar.header("🕒 Time Info")
month = st.sidebar.slider("Month", min_value=1, max_value=12, value=6)
day = st.sidebar.slider("Day", min_value=1, max_value=31, value=15)
hour = st.sidebar.slider("Hour", min_value=0, max_value=23, value=12)

# প্রেডিক্ট বাটন
if st.button("Predict Energy Consumption"):
    # ৪. ইউজারের ইনপুট দিয়ে ডিকশনারি তৈরি
    input_data = {
        'square_feet': square_feet,
        'year_built': year_built,
        'floor_count': floor_count,
        'air_temperature': air_temperature,
        'cloud_coverage': cloud_coverage,
        'hour': hour,
        'day': day,
        'month': month,
        
        # বাকি ফিচারগুলো ডিফল্ট ভ্যালু (যাতে এরর না আসে)
        'building_id': 0,
        'meter': 0,
        'site_id': 0,
        'dew_temperature': 20.0,
        'precip_depth_1_hr': 0.0,
        'sea_level_pressure': 1019.0,
        'wind_direction': 0.0,
        'wind_speed': 0.0
    }

    input_df = pd.DataFrame([input_data])

    # ৫. One-Hot Encoding (primary_use এর জন্য)
    # প্রথমে primary_use রিলেটেড সব কলামে 0 বসাই
    for col in model_columns:
        if 'primary_use_' in str(col):
            input_df[col] = 0
    
    selected_use_col = f'primary_use_{primary_use}'
    if selected_use_col in model_columns:
        input_df[selected_use_col] = 1

    # ৬. মডেলের কলাম সিরিয়াল অনুযায়ী ডাটাফ্রেম ঠিক করা
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # ঠিক নোটবুকের অর্ডারে কলামগুলো সাজিয়ে নেওয়া
    input_df = input_df[model_columns]

    # ৭. প্রেডিকশন
    # নোটবুকে np.log1p ব্যবহার করা হয়েছিল, তাই এখানে np.expm1 দিয়ে আসল ভ্যালুতে রূপান্তর করতে হবে
    pred_log = model.predict(input_df)[0]
    pred_actual = np.expm1(pred_log) 

    # ৮. ফলাফল দেখানো
    st.success(f"📊 Predicted Meter Reading: **{pred_actual:.2f}**")
