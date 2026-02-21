import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------
# Load Rust Model
# -------------------------
rust_model = tf.saved_model.load("model.rust")
rust_infer = rust_model.signatures["serving_default"]

rust_class_names = [
    line.strip()
    for line in open("model.rust/labels.txt", "r")
]

# -------------------------
# Load Seat Model
# -------------------------
seat_model = tf.saved_model.load("model.seat")
seat_infer = seat_model.signatures["serving_default"]

seat_class_names = [
    line.strip()
    for line in open("model.seat/labels.txt", "r")
]
# -------------------------
# Load Mudguard Model
# -------------------------
mudguard_model = tf.saved_model.load("model.mudguard")
mudguard_infer = mudguard_model.signatures["serving_default"]

mudguard_class_names = [
    line.strip()
    for line in open("model.mudguard/labels.txt", "r")
]

st.title("🚲 Cycle Resale Price Estimator")

mrp = st.number_input("Enter MRP (₹)", min_value=0.0)
years_used = st.number_input("Years Used", min_value=0)
months_used = st.number_input("Additional Months Used", min_value=0, max_value=11)

rust_file = st.file_uploader(
    "Upload Full Cycle Image (Rust Detection)",
    key="rust",
    type=["jpg", "jpeg", "png"]
)

seat_file = st.file_uploader(
    "Upload Seat Close-up Image (Seat Damage Detection)",
    key="seat",
    type=["jpg", "jpeg", "png"]
)
mudguard_file = st.file_uploader(
    "Upload Mudguard Close-up Image (Mudguard Detection)",
    key="mudguard",
    type=["jpg", "jpeg", "png"]
)

rust_confidence = 0
seat_confidence_score = 0
mudguard_confidence_score = 0

# -------------------------
# Rust Prediction
# -------------------------
if rust_file is not None:
    image = Image.open(rust_file).convert("RGB")
    st.image(image, caption="Full Cycle Image")

    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    input_tensor = tf.convert_to_tensor(normalized[np.newaxis, ...])

    prediction = rust_infer(input_tensor)
    prediction_values = list(prediction.values())[0].numpy()

    index = np.argmax(prediction_values)
    predicted_label = rust_class_names[index]
    confidence_score = float(prediction_values[0][index])

    st.write("Rust Prediction:", predicted_label)
    st.write("Confidence:", round(confidence_score * 100, 2), "%")

    if index == 0:  # Ensure class 0 = Rust
        rust_confidence = confidence_score


# -------------------------
# Seat Prediction
# -------------------------
if seat_file is not None:
    seat_image = Image.open(seat_file).convert("RGB")
    st.image(seat_image, caption="Seat Image")

    seat_image = seat_image.resize((224, 224))
    seat_array = np.asarray(seat_image)
    seat_normalized = (seat_array.astype(np.float32) / 127.5) - 1
    seat_tensor = tf.convert_to_tensor(seat_normalized[np.newaxis, ...])

    seat_prediction = seat_infer(seat_tensor)
    seat_values = list(seat_prediction.values())[0].numpy()

    seat_index = np.argmax(seat_values)
    seat_label = seat_class_names[seat_index]
    seat_confidence = float(seat_values[0][seat_index])

    st.write("Seat Prediction:", seat_label)
    st.write("Seat Confidence:", round(seat_confidence * 100, 2), "%")

    if seat_index == 0:  # Ensure class 0 = Damaged Seat
        seat_confidence_score = seat_confidence


if mudguard_file is not None:
    mudguard_image = Image.open(mudguard_file).convert("RGB")
    st.image(mudguard_image, caption="Mudguard Image")

    mudguard_image = mudguard_image.resize((224, 224))
    mudguard_array = np.asarray(mudguard_image)
    mudguard_normalized = (mudguard_array.astype(np.float32) / 127.5) - 1
    mudguard_tensor = tf.convert_to_tensor(mudguard_normalized[np.newaxis, ...])

    mudguard_prediction = mudguard_infer(mudguard_tensor)
    mudguard_values = list(mudguard_prediction.values())[0].numpy()

    mudguard_index = np.argmax(mudguard_values)
    mudguard_label = mudguard_class_names[mudguard_index]
    mudguard_confidence = float(mudguard_values[0][mudguard_index])

    st.write("Mudguard Prediction:", mudguard_label)
    st.write("Mudguard Confidence:", round(mudguard_confidence * 100, 2), "%")

    # Assuming class 0 = Missing Mudguard
    if mudguard_index == 0:
        mudguard_confidence_score = mudguard_confidence

# -------------------------
# Price Calculation
# -------------------------
if st.button("Calculate Final Price"):

    total_months = years_used * 12 + months_used
    annual_rate = 0.15
    monthly_rate = annual_rate / 12

    depreciation_factor = (1 - monthly_rate) ** total_months
    depreciated_value = mrp * depreciation_factor

    rust_deduction = 500 * rust_confidence
    seat_deduction = 400 * seat_confidence_score
    mudguard_deduction = 250 * mudguard_confidence_score

    final_price = max(
        depreciated_value - rust_deduction - seat_deduction - mudguard_deduction,
        0
    )

    st.write("Value after depreciation (15% Per Year): ₹", round(depreciated_value, 2))
    st.write("Rust Deduction: ₹", round(rust_deduction, 2))
    st.write("This model deducts ₹ 500 Per cycle if rust is detected * confidence %")
    st.write("Seat Deduction: ₹", round(seat_deduction, 2))
    st.write("This model deducts ₹ 400 Per cycle if seat damage is detected * confidence %")
    st.write("Mudguard Deduction: ₹", round(mudguard_deduction, 2))
    st.write("This model deducts ₹ 250 Per cycle if mudguard is not detected * confidence %")


    st.success(f"Final Estimated Price: ₹ {round(final_price, 2)}")
