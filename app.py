import streamlit as st
import pickle

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered"
)

# =====================================================
# LOAD MODEL
# =====================================================

with open("spam_model (2).pkl", "rb") as f:
    model = pickle.load(f)

# =====================================================
# LOAD TF-IDF VECTORIZER
# =====================================================

with open("tfidf_vectorizer (1).pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =====================================================
# TITLE
# =====================================================

st.title("🛡️ SpamShield AI")

st.write(
    "Enter a message below and the AI will detect "
    "whether it is Spam or Safe."
)

# =====================================================
# MESSAGE INPUT
# =====================================================

message = st.text_area(
    "📩 Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You won ₹50,000. Click here to claim."
)

# =====================================================
# PREDICTION
# =====================================================

if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")

    else:

        # Convert message into TF-IDF
        message_vector = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vector)[0]

        # 0 = Safe
        # 1 = Spam

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message appears to be spam.")

        else:
            st.success("✅ SAFE MESSAGE")
            st.write("This message appears to be safe.")

# =====================================================
# EXAMPLES
# =====================================================

st.divider()

st.subheader("💡 Example Messages")

st.write("🚨 Spam:")
st.code(
    "Congratulations! You have won ₹50,000. "
    "Click here to claim your prize!"
)

st.write("✅ Safe:")
st.code(
    "Hey, are you coming to college tomorrow?"
)