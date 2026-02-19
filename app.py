import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load SavedModel
model = tf.saved_model.load("model.rust")
infer = model.signatures["serving_default"]

class_names = [line.strip() for line in open("model.rust/labels.txt", "r").readlines()]

st.title("🚲 Cycle Rust Detector")

mrp = st.number_input("Enter MRP (₹)", min_value=0.0)
years_used = st.number_input("Years Used", min_value=0)
months_used = st.number_input("Additional Months Used", min_value=0, max_value=11)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

rust_confidence = 0

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    input_tensor = tf.convert_to_tensor(normalized[np.newaxis, ...])

    prediction = infer(input_tensor)
    prediction_values = list(prediction.values())[0].numpy()

    index = np.argmax(prediction_values)
    predicted_label = class_names[index]
    confidence_score = float(prediction_values[0][index])

    st.write("Prediction:", predicted_label)
    st.write("Confidence:", round(confidence_score * 100, 2), "%")

    # Apply deduction only if rust class
    if index == 0:   # Change if rust class index is different
        rust_confidence = confidence_score
    else:
        rust_confidence = 0


if st.button("Calculate Final Price"):
    total_months = years_used * 12 + months_used
    annual_rate = 0.15
    monthly_rate = annual_rate / 12

    depreciation_factor = (1 - monthly_rate) ** total_months
    depreciated_value = mrp * depreciation_factor

    rust_deduction = 500 * rust_confidence
    final_price = max(depreciated_value - rust_deduction, 0)

    st.write("Value after depreciation: ₹", round(depreciated_value, 2))
    st.write("Rust Deduction: ₹", round(rust_deduction, 2))
    st.success(f"Final Estimated Price: ₹ {round(final_price, 2)}")
