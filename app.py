import streamlit as st
import joblib



model = joblib.load("model.pkl")



st.set_page_config(
    page_title="AI Cybersecurity Assistant",
    page_icon="🛡️",
    layout="centered"
)


st.title("🛡️ AI Cybersecurity Assistant")
st.subheader("AI-Based Phishing Detection")



st.write(
    "Enter an email or message below to check whether it "
    "is Safe or Phishing."
)



message = st.text_area(
    "Enter Email / Message",
    height=180,
    placeholder="Paste your email or message here..."
)



if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        
        prediction = model.predict([message])[0]

        
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100

        st.divider()

        if prediction == "phishing":
            st.error("🚨 PHISHING DETECTED")

        else:
            st.success("✅ SAFE MESSAGE")

        st.write(f"**Prediction:** {prediction.upper()}")
        st.write(f"**Confidence:** {confidence:.2f}%")