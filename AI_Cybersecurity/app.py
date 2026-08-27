# AI_Cybersecurity/app.py
# UI-only improvements for a college-friendly demo.
# Models, prediction logic, and anomaly detection are unchanged.

import io
import os
import joblib
import pandas as pd
import streamlit as st
from email import policy
from email.parser import BytesParser

# Optional PDF support (no change to ML logic)
try:
    from PyPDF2 import PdfReader
    _has_pdf = True
except Exception:
    _has_pdf = False

APP_DIR = os.path.dirname(__file__) or "."
PHISHING_MODEL_PATH = os.path.join(APP_DIR, "model.pkl")
ANOMALY_MODEL_PATH = os.path.join(APP_DIR, "anomaly_model.pkl")

st.set_page_config(page_title="AI Cybersecurity Assistant", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Cybersecurity Assistant")

# -------------------------
# Helpers & model loading (unchanged logic)
# -------------------------
def safe_load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Failed to load model at {path}: {e}")
            return None
    return None

def extract_text_from_pdf_bytes(b: bytes) -> str:
    if not _has_pdf:
        return ""
    try:
        reader = PdfReader(io.BytesIO(b))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception:
        return ""

def extract_text_from_eml_bytes(b: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(b)
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    txt = part.get_content()
                    if txt:
                        parts.append(txt)
        else:
            txt = msg.get_content()
            if txt:
                parts.append(txt)
        return "\n".join(parts)
    except Exception:
        return ""

def get_probability_for_model(model, texts):
    # Keep the same approach: prefer predict_proba, else decision_function mapped via sigmoid.
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(texts)
            # If binary classification, return probability of positive (index 1)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 0]
            return probs.max(axis=1)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            import numpy as np
            df = model.decision_function(texts)
            df = np.array(df, dtype=float)
            probs_pos = 1.0 / (1.0 + np.exp(-df))
            return probs_pos
        except Exception:
            pass
    # fallback
    return [0.0] * len(texts)

@st.cache_resource
def load_phishing_model():
    return safe_load_model(PHISHING_MODEL_PATH)

@st.cache_resource
def load_anomaly_model():
    return safe_load_model(ANOMALY_MODEL_PATH)

phishing_model = load_phishing_model()
anomaly_model = load_anomaly_model()

# Session counters (for dashboard metrics)
if "messages_checked" not in st.session_state:
    st.session_state["messages_checked"] = 0
if "phishing_detected" not in st.session_state:
    st.session_state["phishing_detected"] = 0
if "activities_checked" not in st.session_state:
    st.session_state["activities_checked"] = 0
if "suspicious_activities" not in st.session_state:
    st.session_state["suspicious_activities"] = 0

# Store last results for advanced view
if "last_phish_results" not in st.session_state:
    st.session_state["last_phish_results"] = None
if "last_anom_results" not in st.session_state:
    st.session_state["last_anom_results"] = None

# -------------------------
# Layout: three tabs
# -------------------------
tabs = st.tabs(["Phishing Detection", "User Behavior Analysis", "Dashboard"])

# -------------------------
# PHISHING DETECTION (UI improved)
# -------------------------
with tabs[0]:
    st.header("Phishing Detection")
    st.write("Paste a message, upload files, or upload a CSV to check if messages are safe or phishing.")
    col_main, col_side = st.columns([3, 1])

    with col_main:
        message = st.text_area("Enter Email / Message", height=220, placeholder="Paste your email or message here...")
        uploaded_csv = st.file_uploader("Upload CSV (column: text) for batch prediction", type=["csv"])
        uploaded_file = st.file_uploader("Upload a file (.txt, .pdf, .eml) to extract and analyze", type=["txt", "pdf", "eml"])

        st.markdown("**Phishing sensitivity (Detection threshold)**")
        st.caption("Higher sensitivity (lower threshold) will label more messages as phishing; a lower sensitivity labels fewer messages as phishing.")
        threshold = st.slider("Detection threshold", 0.0, 1.0, 0.50)

        analyze_btn = st.button("🔍 Analyze")

        if analyze_btn:
            if phishing_model is None:
                st.warning("Phishing model not found. Please ensure AI_Cybersecurity/model.pkl exists.")
            else:
                texts = []
                sources = []

                # typed message
                if message and message.strip():
                    texts.append(message.strip())
                    sources.append("typed")

                # uploaded file
                if uploaded_file is not None:
                    fname = uploaded_file.name.lower()
                    b = uploaded_file.getvalue()
                    if fname.endswith(".txt"):
                        try:
                            txt = b.decode("utf-8", errors="ignore")
                        except Exception:
                            txt = str(b)
                        texts.append(txt)
                        sources.append(uploaded_file.name)
                    elif fname.endswith(".pdf"):
                        pdf_text = extract_text_from_pdf_bytes(b)
                        if pdf_text:
                            texts.append(pdf_text)
                            sources.append(uploaded_file.name)
                        else:
                            st.warning("Could not extract text from PDF (install PyPDF2 or use a text file).")
                    elif fname.endswith(".eml"):
                        eml_text = extract_text_from_eml_bytes(b)
                        if eml_text:
                            texts.append(eml_text)
                            sources.append(uploaded_file.name)
                        else:
                            st.warning("Could not extract text from EML file.")

                # uploaded CSV
                if uploaded_csv is not None:
                    try:
                        df = pd.read_csv(uploaded_csv)
                        if "text" not in df.columns:
                            st.error("CSV must contain a 'text' column.")
                        else:
                            for i, row in df.iterrows():
                                texts.append(str(row["text"]))
                                sources.append(f"csv_row_{i}")
                    except Exception as e:
                        st.error(f"Failed to read CSV: {e}")

                if len(texts) == 0:
                    st.info("Provide text (type, file, or CSV) to analyze.")
                else:
                    # call existing prediction logic (unchanged)
                    try:
                        preds = phishing_model.predict(texts)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        preds = ["error"] * len(texts)
                    probs = get_probability_for_model(phishing_model, texts)

                    # Build user-friendly display
                    display_rows = []
                    detected_count = 0
                    for src, p, prob in zip(sources, preds, probs):
                        prob_val = float(prob)
                        is_phish = prob_val >= threshold
                        label_text = "🔴 PHISHING DETECTED" if is_phish else "🟢 SAFE MESSAGE"
                        confidence_value = prob_val if is_phish else (1 - prob_val)
                        confidence_pct = f"{confidence_value*100:.2f}%"

                        display_rows.append({
                            "source": src,
                            "label_text": label_text,
                            "confidence": confidence_pct,
                            "is_phish": is_phish,
                            # keep raw data for advanced view
                            "model_raw": str(p),
                            "prob_phishing": float(prob)
                        })
                        # update session counters
                        st.session_state["messages_checked"] += 1
                        if is_phish:
                            st.session_state["phishing_detected"] += 1
                            detected_count += 1

                    # Show the MAIN result(s) prominently
                    for row in display_rows:
                        st.markdown("---")
                        if row["is_phish"]:
                            st.markdown(f"### 🔴 PHISHING DETECTED")
                            st.markdown(f"**Confidence:** {row['confidence']}")
                            st.error("This message looks like phishing. Do not click links or provide sensitive info.")
                        else:
                            st.markdown(f"### 🟢 SAFE MESSAGE")
                            st.markdown(f"**Confidence:** {row['confidence']}")
                            st.success("This message appears safe.")

                    # Save last results for Dashboard and advanced view
                    df_results = pd.DataFrame(display_rows)
                    st.session_state["last_phish_results"] = df_results

                    # Advanced / Technical details (hidden by default)
                    with st.expander("Advanced Details / Technical Information (optional)"):
                        st.caption("Raw model outputs and sources (for reviewers / debugging).")
                        st.dataframe(df_results[["source", "model_raw", "prob_phishing"]], use_container_width=True)

    with col_side:
        st.markdown("**Model Status**")
        if phishing_model is not None:
            st.success("🟢 Phishing model: READY")
        else:
            st.error("🔴 Phishing model: MISSING")

# -------------------------
# USER BEHAVIOR ANALYSIS (UI improved)
# -------------------------
with tabs[1]:
    st.header("User Behavior Analysis")
    st.write("Upload a CSV containing user activity features to check for suspicious behavior.")
    st.markdown("Required numeric columns: login_hour, login_duration, failed_attempts, device_change, location_change")
    col_up, col_action = st.columns([3, 1])

    with col_up:
        uploaded_activity = st.file_uploader("Upload user_activity CSV", type=["csv"], key="activity_ui")
        score_btn = st.button("⚡ Score Activity")

    with col_action:
        st.markdown("**Model Status**")
        if anomaly_model is not None:
            st.success("🟢 Behavior model: READY")
        else:
            st.error("🔴 Behavior model: MISSING")

    if score_btn:
        if uploaded_activity is None:
            st.info("Please upload a CSV file first.")
        else:
            try:
                df_act = pd.read_csv(uploaded_activity)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df_act = None

            if df_act is not None:
                required_cols = ["login_hour", "login_duration", "failed_attempts", "device_change", "location_change"]
                missing = [c for c in required_cols if c not in df_act.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    if anomaly_model is None:
                        st.warning("Anomaly model not found. Ensure AI_Cybersecurity/anomaly_model.pkl exists.")
                    else:
                        X = df_act[required_cols].values
                        # use existing scoring/prediction logic (unchanged)
                        try:
                            scores = anomaly_model.score_samples(X)
                        except Exception as e:
                            st.error(f"Scoring failed: {e}")
                            scores = [0.0] * len(X)
                        try:
                            preds = anomaly_model.predict(X)
                        except Exception:
                            preds = None

                        # Map prediction outputs into a boolean 'suspicious' array without changing algorithm
                        import numpy as np
                        suspicious_mask = None
                        if preds is not None:
                            arr = np.array(preds)
                            uniques = set(arr.tolist())
                            if -1 in uniques and 1 in uniques:
                                suspicious_mask = arr == -1
                            elif set(uniques).issubset({0, 1}):
                                suspicious_mask = arr == 1
                            elif set(uniques).issubset({True, False}):
                                suspicious_mask = arr.astype(bool)
                        # If mapping failed, fallback to using scores (lower score => more anomalous)
                        if suspicious_mask is None:
                            try:
                                # Mark bottom 5% by score as suspicious (fallback only)
                                thr = np.percentile(scores, 5)
                                suspicious_mask = np.array(scores) < thr
                            except Exception:
                                suspicious_mask = np.array([False] * len(scores))

                        # Build result table
                        df_out = df_act.copy()
                        df_out["anomaly_score"] = scores
                        df_out["is_suspicious"] = suspicious_mask

                        # update session counters
                        st.session_state["activities_checked"] += len(df_out)
                        st.session_state["suspicious_activities"] += int(suspicious_mask.sum())

                        # Summary for normal users (clean)
                        total = len(df_out)
                        normal_count = int(total - suspicious_mask.sum())
                        suspicious_count = int(suspicious_mask.sum())

                        st.markdown("### 👤 USER BEHAVIOR RESULT")
                        st.metric("Activities Checked", value=total)
                        st.write("")
                        coln, cols = st.columns(2)
                        with coln:
                            st.markdown(f"🟢 Normal Activities: **{normal_count}**")
                        with cols:
                            st.markdown(f"🔴 Suspicious Activities: **{suspicious_count}**")

                        if suspicious_count > 0:
                            st.warning("⚠️ Suspicious activity detected")
                        else:
                            st.success("✅ No suspicious activity detected")

                        # Save detailed results for dashboard / download
                        st.session_state["last_anom_results"] = df_out

                        # Detailed results hidden in expander
                        with st.expander("View Detailed Results"):
                            st.caption("Detailed per-row results (anomaly_score and is_suspicious)")
                            st.dataframe(df_out, use_container_width=True)
                            flagged = df_out[df_out["is_suspicious"]]
                            if not flagged.empty:
                                st.download_button(
                                    "Download flagged rows as CSV",
                                    flagged.to_csv(index=False).encode("utf-8"),
                                    file_name="flagged_user_activity.csv"
                                )

# -------------------------
# DASHBOARD (UI improved)
# -------------------------
with tabs[2]:
    st.header("Dashboard")
    st.write("Quick overview of model status and recent activity (session only).")

    # Model status cards
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        if phishing_model is not None:
            st.markdown("### 🟢 Phishing Detection")
            st.write("Status: READY")
        else:
            st.markdown("### 🔴 Phishing Detection")
            st.write("Status: MISSING")
    with mcol2:
        if anomaly_model is not None:
            st.markdown("### 🟢 User Behavior")
            st.write("Status: READY")
        else:
            st.markdown("### 🔴 User Behavior")
            st.write("Status: MISSING")

    st.markdown("---")

    # Four large metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="📩 Messages Checked", value=st.session_state["messages_checked"])
    c2.metric(label="🚨 Phishing Detected", value=st.session_state["phishing_detected"])
    c3.metric(label="👤 Activities Checked", value=st.session_state["activities_checked"])
    c4.metric(label="⚠️ Suspicious Activities", value=st.session_state["suspicious_activities"])

    st.markdown("---")

    # Simple summary with emojis and big numbers
    st.subheader("Summary")
    last_phish = st.session_state.get("last_phish_results")
    if last_phish is not None:
        total_msgs = len(last_phish)
        safe_msgs = int((last_phish["label_text"] == "🟢 SAFE MESSAGE").sum())
        phish_msgs = total_msgs - safe_msgs
    else:
        total_msgs = st.session_state["messages_checked"]
        safe_msgs = max(0, st.session_state["messages_checked"] - st.session_state["phishing_detected"])
        phish_msgs = st.session_state["phishing_detected"]

    last_anom = st.session_state.get("last_anom_results")
    if last_anom is not None:
        total_act = len(last_anom)
        suspicious_act = int(last_anom["is_suspicious"].sum())
        normal_act = total_act - suspicious_act
    else:
        total_act = st.session_state["activities_checked"]
        suspicious_act = st.session_state["suspicious_activities"]
        normal_act = max(0, total_act - suspicious_act)

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"🟢 Safe Messages\n\n**{safe_msgs}**")
    s2.markdown(f"🔴 Phishing Messages\n\n**{phish_msgs}**")
    s3.markdown(f"🟢 Normal Activities\n\n**{normal_act}**")
    s4.markdown(f"🟠 Suspicious Activities\n\n**{suspicious_act}**")

    st.markdown("---")
    st.caption("Metrics reflect activity during the current Streamlit session. To persist metrics, integrate a backend or database for logging.")

# End of UI-only improved app.py