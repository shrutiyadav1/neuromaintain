import keras
import keras.src.ops as internal_ops

# Work around the missing internal Keras function
internal_ops.zeros = keras.ops.zeros
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.load_weights("neuromaintain_model.keras")

model.load_weights("neuromaintain_model.keras")
scaler = joblib.load("neuromaintain_scaler.pkl")

st.set_page_config(page_title="NeuroMaintain", page_icon="⚙️")

st.title("⚙️ NeuroMaintain")
st.subheader("Intelligent Machine Failure Prediction")

st.write(
    "Enter the machine sensor readings to estimate "
    "the probability of machine failure."
)

# Sensor inputs
air_temp = st.number_input("Air Temperature (K)", 290.0, 310.0, 300.0)
process_temp = st.number_input("Process Temperature (K)", 300.0, 320.0, 310.0)
rot_speed = st.number_input("Rotational Speed (rpm)", 500, 3000, 1500)
torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
tool_wear = st.number_input("Tool Wear (min)", 0, 300, 100)

machine_type = st.selectbox("Machine Type", ["H", "L", "M"])

if st.button("Predict Machine Failure"):
    new_machine = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": int(machine_type == "H"),
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M")
    }])

    new_machine_scaled = scaler.transform(new_machine)
    probability = float(model.predict(new_machine_scaled, verbose=0)[0][0])

    st.metric("Estimated Failure Probability", f"{probability:.2%}")

    if probability >= 0.5:
        st.error("⚠️ Failure Predicted")
    else:
        st.success("✅ No Failure Predicted")

    st.caption(
        "This is a model estimate, not a guarantee of machine condition."
    )
