import streamlit as st
import pickle
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Wine Classification",
    page_icon="🍷",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    # Get the folder where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "model_rf.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    # Check model file
    if not os.path.exists(model_path):
        st.error("❌ model_rf.pkl file not found!")
        st.info(
            "Please keep model_rf.pkl in the same folder as app.py."
        )
        st.stop()

    # Check scaler file
    if not os.path.exists(scaler_path):
        st.error("❌ scaler.pkl file not found!")
        st.info(
            "Please keep scaler.pkl in the same folder as app.py."
        )
        st.stop()

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🍷 Wine Classification")

st.write(
    "Enter the wine features below to predict the wine class."
)

# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

st.subheader("Enter Wine Features")

alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    max_value=20.0,
    value=13.0
)

malic_acid = st.number_input(
    "Malic Acid",
    min_value=0.0,
    max_value=10.0,
    value=2.0
)

ash = st.number_input(
    "Ash",
    min_value=0.0,
    max_value=5.0,
    value=2.3
)

alcalinity_of_ash = st.number_input(
    "Alcalinity of Ash",
    min_value=0.0,
    max_value=40.0,
    value=19.0
)

magnesium = st.number_input(
    "Magnesium",
    min_value=0.0,
    max_value=200.0,
    value=100.0
)

total_phenols = st.number_input(
    "Total Phenols",
    min_value=0.0,
    max_value=10.0,
    value=2.5
)

flavanoids = st.number_input(
    "Flavanoids",
    min_value=0.0,
    max_value=10.0,
    value=2.5
)

nonflavanoid_phenols = st.number_input(
    "Nonflavanoid Phenols",
    min_value=0.0,
    max_value=5.0,
    value=0.3
)

proanthocyanins = st.number_input(
    "Proanthocyanins",
    min_value=0.0,
    max_value=5.0,
    value=1.5
)

color_intensity = st.number_input(
    "Color Intensity",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

hue = st.number_input(
    "Hue",
    min_value=0.0,
    max_value=5.0,
    value=1.0
)

od280_od315 = st.number_input(
    "OD280/OD315",
    min_value=0.0,
    max_value=5.0,
    value=3.0
)

proline = st.number_input(
    "Proline",
    min_value=0.0,
    max_value=2000.0,
    value=700.0
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Wine Class"):

    # Create input data
    input_data = [[
        alcohol,
        malic_acid,
        ash,
        alcalinity_of_ash,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        od280_od315,
        proline
    ]]

    try:

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Display result
        st.success(
            f"🍷 Predicted Wine Class: {prediction[0]}"
        )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.write("Error:", e)