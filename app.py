import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "pneumonia_classifier.keras"
    )


model = load_model()


# -----------------------------
# Grad-CAM
# -----------------------------

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="last_conv"
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        class_score = predictions[:, 0]

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# -----------------------------
# Website title
# -----------------------------

st.title("🩺 AI-Based Chest X-Ray Classifier")

st.write(
    "Upload a chest X-ray image to classify it as "
    "Normal or Pneumonia."
)

st.warning(
    "Educational project only. "
    "This system is not a medical diagnostic tool."
)


# -----------------------------
# Upload image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        width=500
    )


    # -------------------------
    # Preprocess
    # -------------------------

    resized_image = image.resize(
        (128, 128)
    )

    img_array = np.array(
        resized_image
    ).astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # -------------------------
    # Prediction
    # -------------------------

    if st.button("🔍 Analyze X-Ray"):

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]


        if prediction >= 0.5:

            predicted_class = "PNEUMONIA"
            confidence = prediction * 100

        else:

            predicted_class = "NORMAL"
            confidence = (1 - prediction) * 100


        st.subheader("AI Result")

        if predicted_class == "PNEUMONIA":

            st.error(
                f"Prediction: {predicted_class}"
            )

        else:

            st.success(
                f"Prediction: {predicted_class}"
            )


        st.info(
            f"Model score: {confidence:.2f}%"
        )


        # -------------------------
        # Grad-CAM
        # -------------------------

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "last_conv"
        )


        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.imshow(
            resized_image
        )

        ax.imshow(
            heatmap,
            alpha=0.5,
            cmap="jet"
        )

        ax.axis("off")

        ax.set_title(
            "Grad-CAM AI Explanation"
        )

        st.pyplot(fig)


st.markdown("---")

st.caption(
    "This application is intended for educational "
    "and research purposes only."
)
