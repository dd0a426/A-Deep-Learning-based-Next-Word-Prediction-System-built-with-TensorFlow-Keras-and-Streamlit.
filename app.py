import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #4F46E5;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 1.1rem;
}

.prediction-box {
    background: linear-gradient(135deg,#4F46E5,#7C3AED);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Resources
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model (1).h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# Create reverse dictionary
index_word = {v: k for k, v in tokenizer.word_index.items()}

# ------------------------------
# Prediction Function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len-1,
        padding='pre'
    )

    preds = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(preds, axis=-1)[0]

    return index_word.get(predicted_index, "Unknown")

# ------------------------------
# Header
# ------------------------------
st.markdown(
    "<div class='big-title'>🧠 Next Word Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Powered by LSTM Deep Learning Model</div>",
    unsafe_allow_html=True
)

st.write("")
st.write("")

# ------------------------------
# Input Section
# ------------------------------
st.subheader("✍️ Enter Your Text")

user_input = st.text_input(
    "",
    placeholder="Example: I love machine"
)

col1, col2 = st.columns([1,1])

with col1:
    predict_btn = st.button("🚀 Predict")

with col2:
    clear_btn = st.button("🗑️ Clear")

# ------------------------------
# Prediction
# ------------------------------
if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")

    else:
        with st.spinner("🔮 Predicting next word..."):
            next_word = predict_next_word(user_input)

        st.markdown(
            f"""
            <div class='prediction-box'>
            Predicted Word <br><br>
            👉 {next_word}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        width=120
    )

    st.title("About")

    st.info(
        """
        This application predicts the next word
        using a trained LSTM Neural Network.

        📌 Deep Learning

        📌 NLP

        📌 TensorFlow

        📌 Streamlit
        """
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.caption(
    "Developed by Abhay Kumar 🚀 | LSTM Next Word Prediction"
)