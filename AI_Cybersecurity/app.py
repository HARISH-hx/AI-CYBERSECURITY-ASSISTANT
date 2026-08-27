# AI_Cybersecurity/app.py
# AI Cybersecurity Assistant
# College-friendly Streamlit application

import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from email import policy
from email.parser import BytesParser


# =========================================================
# Optional PDF support
# =========================================================
try:
    from PyPDF2 import PdfReader
    _has_pdf = True
except Exception:
    _has_pdf = False


# =========================================================
# File paths
# =========================================================
APP_DIR = os.path.dirname(__file__) or "."

PHISHING_MODEL_PATH = os.path.join(
    APP_DIR,
    "model.pkl"
)

ANOMALY_MODEL_PATH = os.path.join(
    APP_DIR,
    "anomaly_model.pkl"
)


# =========================================================
# Streamlit page settings
# =========================================================
st.set_page_config(
    page_title="AI Cybersecurity Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Cybersecurity Assistant")


# =========================================================
# MODEL LOADING
# =========================================================
def safe_load_model(path):

    if os.path.exists(path):

        try:
            return joblib.load(path)

        except Exception as e:

            st.error(
                f"Failed to load model at {path}: {e}"
            )

            return None

    return None


@st.cache_resource
def load_phishing_model():

    return safe_load_model(
        PHISHING_MODEL_PATH
    )


@st.cache_resource
def load_anomaly_model():

    return safe_load_model(
        ANOMALY_MODEL_PATH
    )


phishing_model = load_phishing_model()
anomaly_model = load_anomaly_model()


# =========================================================
# FILE TEXT EXTRACTION
# =========================================================
def extract_text_from_pdf_bytes(b):

    if not _has_pdf:
        return ""

    try:

        reader = PdfReader(
            io.BytesIO(b)
        )

        parts = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                parts.append(text)

        return "\n".join(parts)

    except Exception:

        return ""


def extract_text_from_eml_bytes(b):

    try:

        msg = BytesParser(
            policy=policy.default
        ).parsebytes(b)

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


# =========================================================
# PHISHING PROBABILITY
# =========================================================
def get_probability_for_model(model, texts):

    """
    Returns phishing probability.

    The Logistic Regression classes are:
    ['phishing', 'safe']

    Therefore phishing probability is column 0.
    """

    if hasattr(model, "predict_proba"):

        try:

            probs = model.predict_proba(texts)

            if (
                probs.ndim == 2
                and probs.shape[1] >= 2
            ):

                classes = list(
                    model.named_steps[
                        "classifier"
                    ].classes_
                )

                if "phishing" in classes:

                    phishing_index = classes.index(
                        "phishing"
                    )

                    return probs[
                        :,
                        phishing_index
                    ]

                return probs[:, 0]

            return probs.max(axis=1)

        except Exception:
            pass


    # Fallback for models using decision_function
    if hasattr(model, "decision_function"):

        try:

            df = model.decision_function(
                texts
            )

            df = np.array(
                df,
                dtype=float
            )

            probs = 1.0 / (
                1.0 + np.exp(-df)
            )

            return probs

        except Exception:
            pass


    return [0.0] * len(texts)


# =========================================================
# SESSION COUNTERS
# =========================================================
if "messages_checked" not in st.session_state:

    st.session_state[
        "messages_checked"
    ] = 0


if "phishing_detected" not in st.session_state:

    st.session_state[
        "phishing_detected"
    ] = 0


if "activities_checked" not in st.session_state:

    st.session_state[
        "activities_checked"
    ] = 0


if "suspicious_activities" not in st.session_state:

    st.session_state[
        "suspicious_activities"
    ] = 0


# =========================================================
# STORE LAST RESULTS
# =========================================================
if "last_phish_results" not in st.session_state:

    st.session_state[
        "last_phish_results"
    ] = None


if "last_anom_results" not in st.session_state:

    st.session_state[
        "last_anom_results"
    ] = None


# =========================================================
# THREE TABS
# =========================================================
tabs = st.tabs(
    [
        "Phishing Detection",
        "User Behavior Analysis",
        "Dashboard"
    ]
)


# =========================================================
# TAB 1
# PHISHING DETECTION
# =========================================================
with tabs[0]:

    st.header(
        "Phishing Detection"
    )

    st.write(
        "Paste a message, upload files, "
        "or upload a CSV to check if "
        "messages are safe or phishing."
    )


    col_main, col_side = st.columns(
        [3, 1]
    )


    # -----------------------------------------------------
    # MAIN COLUMN
    # -----------------------------------------------------
    with col_main:

        message = st.text_area(
            "Enter Email / Message",
            height=220,
            placeholder=(
                "Paste your email or message here..."
            )
        )


        uploaded_csv = st.file_uploader(
            "Upload CSV (column: text) for batch prediction",
            type=["csv"]
        )


        uploaded_file = st.file_uploader(
            "Upload a file (.txt, .pdf, .eml) "
            "to extract and analyze",
            type=["txt", "pdf", "eml"]
        )


        st.markdown(
            "**Phishing sensitivity "
            "(Detection threshold)**"
        )


        st.caption(
            "Higher sensitivity (lower threshold) "
            "will label more messages as phishing; "
            "a higher threshold labels fewer messages "
            "as phishing."
        )


        threshold = st.slider(
            "Detection threshold",
            0.0,
            1.0,
            0.50
        )


        analyze_btn = st.button(
            "🔍 Analyze"
        )


        # -------------------------------------------------
        # ANALYZE PHISHING
        # -------------------------------------------------
        if analyze_btn:

            if phishing_model is None:

                st.warning(
                    "Phishing model not found. "
                    "Please ensure model.pkl exists."
                )

            else:

                texts = []
                sources = []


                # -----------------------------------------
                # Typed message
                # -----------------------------------------
                if message and message.strip():

                    texts.append(
                        message.strip()
                    )

                    sources.append(
                        "typed"
                    )


                # -----------------------------------------
                # Uploaded TXT / PDF / EML
                # -----------------------------------------
                if uploaded_file is not None:

                    fname = (
                        uploaded_file.name.lower()
                    )

                    b = uploaded_file.getvalue()


                    if fname.endswith(".txt"):

                        try:

                            txt = b.decode(
                                "utf-8",
                                errors="ignore"
                            )

                        except Exception:

                            txt = str(b)


                        texts.append(txt)

                        sources.append(
                            uploaded_file.name
                        )


                    elif fname.endswith(".pdf"):

                        pdf_text = (
                            extract_text_from_pdf_bytes(
                                b
                            )
                        )


                        if pdf_text:

                            texts.append(
                                pdf_text
                            )

                            sources.append(
                                uploaded_file.name
                            )

                        else:

                            st.warning(
                                "Could not extract text "
                                "from PDF."
                            )


                    elif fname.endswith(".eml"):

                        eml_text = (
                            extract_text_from_eml_bytes(
                                b
                            )
                        )


                        if eml_text:

                            texts.append(
                                eml_text
                            )

                            sources.append(
                                uploaded_file.name
                            )

                        else:

                            st.warning(
                                "Could not extract text "
                                "from EML file."
                            )


                # -----------------------------------------
                # Uploaded CSV
                # -----------------------------------------
                if uploaded_csv is not None:

                    try:

                        df = pd.read_csv(
                            uploaded_csv
                        )


                        if "text" not in df.columns:

                            st.error(
                                "CSV must contain "
                                "a 'text' column."
                            )

                        else:

                            for i, row in df.iterrows():

                                texts.append(
                                    str(row["text"])
                                )

                                sources.append(
                                    f"csv_row_{i}"
                                )


                    except Exception as e:

                        st.error(
                            f"Failed to read CSV: {e}"
                        )


                # -----------------------------------------
                # No input
                # -----------------------------------------
                if len(texts) == 0:

                    st.info(
                        "Provide text (type, file, "
                        "or CSV) to analyze."
                    )


                else:

                    # -------------------------------------
                    # Model prediction
                    # -------------------------------------
                    try:

                        preds = phishing_model.predict(
                            texts
                        )

                    except Exception as e:

                        st.error(
                            f"Prediction failed: {e}"
                        )

                        preds = [
                            "error"
                        ] * len(texts)


                    probs = get_probability_for_model(
                        phishing_model,
                        texts
                    )


                    # -------------------------------------
                    # Build display results
                    # -------------------------------------
                    display_rows = []

                    detected_count = 0


                    for src, p, prob in zip(
                        sources,
                        preds,
                        probs
                    ):

                        prob_val = float(prob)


                        # Probability threshold
                        is_phish = (
                            prob_val >= threshold
                        )


                        if is_phish:

                            label_text = (
                                "🔴 PHISHING DETECTED"
                            )

                            confidence_value = (
                                prob_val
                            )

                        else:

                            label_text = (
                                "🟢 SAFE MESSAGE"
                            )

                            confidence_value = (
                                1 - prob_val
                            )


                        confidence_pct = (
                            f"{confidence_value * 100:.2f}%"
                        )


                        display_rows.append(
                            {
                                "source": src,
                                "label_text": label_text,
                                "confidence": confidence_pct,
                                "is_phish": is_phish,
                                "model_raw": str(p),
                                "prob_phishing": prob_val
                            }
                        )


                        # Dashboard counters
                        st.session_state[
                            "messages_checked"
                        ] += 1


                        if is_phish:

                            st.session_state[
                                "phishing_detected"
                            ] += 1

                            detected_count += 1


                    # -------------------------------------
                    # Display main result
                    # -------------------------------------
                    for row in display_rows:

                        st.markdown("---")


                        if row["is_phish"]:

                            st.markdown(
                                "### 🔴 PHISHING DETECTED"
                            )

                            st.markdown(
                                f"**Confidence:** "
                                f"{row['confidence']}"
                            )

                            st.error(
                                "This message looks like "
                                "phishing. Do not click links "
                                "or provide sensitive information."
                            )


                        else:

                            st.markdown(
                                "### 🟢 SAFE MESSAGE"
                            )

                            st.markdown(
                                f"**Confidence:** "
                                f"{row['confidence']}"
                            )

                            st.success(
                                "This message appears safe."
                            )


                    # -------------------------------------
                    # Save phishing results
                    # -------------------------------------
                    df_results = pd.DataFrame(
                        display_rows
                    )


                    st.session_state[
                        "last_phish_results"
                    ] = df_results


                    # -------------------------------------
                    # Technical details
                    # -------------------------------------
                    with st.expander(
                        "Technical Details"
                    ):

                        st.caption(
                            "Technical model information "
                            "for debugging or demonstration."
                        )


                        st.dataframe(
                            df_results[
                                [
                                    "source",
                                    "model_raw",
                                    "prob_phishing"
                                ]
                            ],
                            use_container_width=True
                        )


    # -----------------------------------------------------
    # SIDE COLUMN
    # -----------------------------------------------------
    with col_side:

        st.markdown(
            "**Model Status**"
        )


        if phishing_model is not None:

            st.success(
                "🟢 Phishing model: READY"
            )

        else:

            st.error(
                "🔴 Phishing model: MISSING"
            )


# =========================================================
# TAB 2
# USER BEHAVIOR ANALYSIS
# =========================================================
with tabs[1]:

    st.header(
        "User Behavior Analysis"
    )


    st.write(
        "Upload a CSV containing user activity "
        "features to check for suspicious behavior."
    )


    st.markdown(
        "Required numeric columns: "
        "login_hour, login_duration, "
        "failed_attempts, device_change, "
        "location_change"
    )


    col_up, col_action = st.columns(
        [3, 1]
    )


    # -----------------------------------------------------
    # Upload column
    # -----------------------------------------------------
    with col_up:

        uploaded_activity = st.file_uploader(
            "Upload user_activity CSV",
            type=["csv"],
            key="activity_ui"
        )


        score_btn = st.button(
            "⚡ Score Activity"
        )


    # -----------------------------------------------------
    # Model status
    # -----------------------------------------------------
    with col_action:

        st.markdown(
            "**Model Status**"
        )


        if anomaly_model is not None:

            st.success(
                "🟢 Behavior model: READY"
            )

        else:

            st.error(
                "🔴 Behavior model: MISSING"
            )


    # -----------------------------------------------------
    # Score activity
    # -----------------------------------------------------
    if score_btn:

        if uploaded_activity is None:

            st.info(
                "Please upload a CSV file first."
            )

        else:

            try:

                df_act = pd.read_csv(
                    uploaded_activity
                )

            except Exception as e:

                st.error(
                    f"Failed to read CSV: {e}"
                )

                df_act = None


            if df_act is not None:

                required_cols = [
                    "login_hour",
                    "login_duration",
                    "failed_attempts",
                    "device_change",
                    "location_change"
                ]


                # -----------------------------------------
                # Check columns
                # -----------------------------------------
                missing = [
                    c
                    for c in required_cols
                    if c not in df_act.columns
                ]


                if missing:

                    st.error(
                        f"Missing required columns: {missing}"
                    )


                else:

                    if anomaly_model is None:

                        st.warning(
                            "Anomaly model not found. "
                            "Ensure anomaly_model.pkl exists."
                        )


                    else:

                        # ---------------------------------
                        # Convert input to numeric
                        # ---------------------------------
                        try:

                            X = df_act[
                                required_cols
                            ].apply(
                                pd.to_numeric,
                                errors="raise"
                            ).values

                        except Exception as e:

                            st.error(
                                "Activity data must contain "
                                "only numeric values in the "
                                "required columns."
                            )

                            X = None


                        if X is not None:

                            # -----------------------------
                            # Isolation Forest score
                            # -----------------------------
                            try:

                                scores = (
                                    anomaly_model.score_samples(
                                        X
                                    )
                                )

                            except Exception as e:

                                st.error(
                                    f"Scoring failed: {e}"
                                )

                                scores = np.zeros(
                                    len(X)
                                )


                            # -----------------------------
                            # Isolation Forest prediction
                            # -----------------------------
                            try:

                                preds = (
                                    anomaly_model.predict(
                                        X
                                    )
                                )

                            except Exception as e:

                                st.error(
                                    f"Prediction failed: {e}"
                                )

                                preds = np.ones(
                                    len(X)
                                )


                            isolation_suspicious = (
                                np.array(preds) == -1
                            )


                            # -----------------------------
                            # Cybersecurity rule
                            # -----------------------------
                            #
                            # Multiple strong risk signals:
                            #
                            # - 5 or more failed attempts
                            # - device changed
                            # - location changed
                            #
                            # This prevents clearly risky
                            # activity from being missed by
                            # the unsupervised model.
                            #
                            rule_suspicious = (
                                (
                                    df_act[
                                        "failed_attempts"
                                    ] >= 5
                                )
                                &
                                (
                                    df_act[
                                        "device_change"
                                    ] == 1
                                )
                                &
                                (
                                    df_act[
                                        "location_change"
                                    ] == 1
                                )
                            )


                           # -----------------------------------------
                            # Final cybersecurity decision
                            # -----------------------------------------

                            security_signal = (
                                (df_act["failed_attempts"] >= 3)
                                |
                                (df_act["device_change"] == 1)
                                |
                                (df_act["location_change"] == 1)
                            ).values

                            strong_security_risk = (
                                (df_act["failed_attempts"] >= 5)
                                &
                                (df_act["device_change"] == 1)
                                &
                                (df_act["location_change"] == 1)
                            ).values

                            suspicious_mask = (
                                strong_security_risk
                                |
                                (
                                    isolation_suspicious
                                    & security_signal
                                )
                            )

                            # -----------------------------
                            # Build output
                            # -----------------------------
                            df_out = df_act.copy()


                            df_out[
                                "anomaly_score"
                            ] = scores


                            df_out[
                                "is_suspicious"
                            ] = suspicious_mask


                            # -----------------------------
                            # Session counters
                            # -----------------------------
                            st.session_state[
                                "activities_checked"
                            ] += len(df_out)


                            st.session_state[
                                "suspicious_activities"
                            ] += int(
                                suspicious_mask.sum()
                            )


                            # -----------------------------
                            # Summary
                            # -----------------------------
                            total = len(
                                df_out
                            )


                            suspicious_count = int(
                                suspicious_mask.sum()
                            )


                            normal_count = (
                                total
                                - suspicious_count
                            )


                            st.markdown(
                                "### 👤 USER BEHAVIOR RESULT"
                            )


                            st.metric(
                                "Activities Checked",
                                value=total
                            )


                            st.write("")


                            coln, cols = st.columns(
                                2
                            )


                            with coln:

                                st.markdown(
                                    f"🟢 Normal Activities: "
                                    f"**{normal_count}**"
                                )


                            with cols:

                                st.markdown(
                                    f"🔴 Suspicious Activities: "
                                    f"**{suspicious_count}**"
                                )


                            if suspicious_count > 0:

                                st.warning(
                                    "⚠️ Suspicious activity detected"
                                )

                            else:

                                st.success(
                                    "✅ No suspicious activity detected"
                                )


                            # -----------------------------
                            # Save results
                            # -----------------------------
                            st.session_state[
                                "last_anom_results"
                            ] = df_out


                            # -----------------------------
                            # Detailed results
                            # -----------------------------
                            with st.expander(
                                "View Detailed Results"
                            ):

                                st.caption(
                                    "Detailed per-row results "
                                    "including anomaly score "
                                    "and security decision."
                                )


                                st.dataframe(
                                    df_out,
                                    use_container_width=True
                                )


                                flagged = df_out[
                                    df_out[
                                        "is_suspicious"
                                    ]
                                ]


                                if not flagged.empty:

                                    st.download_button(
                                        "Download flagged rows as CSV",
                                        flagged.to_csv(
                                            index=False
                                        ).encode(
                                            "utf-8"
                                        ),
                                        file_name=(
                                            "flagged_user_activity.csv"
                                        )
                                    )


# =========================================================
# TAB 3
# DASHBOARD
# =========================================================
with tabs[2]:

    st.header(
        "Dashboard"
    )


    st.write(
        "Quick overview of model status and "
        "recent activity (session only)."
    )


    # -----------------------------------------------------
    # Model status
    # -----------------------------------------------------
    mcol1, mcol2 = st.columns(
        2
    )


    with mcol1:

        if phishing_model is not None:

            st.markdown(
                "### 🟢 Phishing Detection"
            )

            st.write(
                "Status: READY"
            )

        else:

            st.markdown(
                "### 🔴 Phishing Detection"
            )

            st.write(
                "Status: MISSING"
            )


    with mcol2:

        if anomaly_model is not None:

            st.markdown(
                "### 🟢 User Behavior"
            )

            st.write(
                "Status: READY"
            )

        else:

            st.markdown(
                "### 🔴 User Behavior"
            )

            st.write(
                "Status: MISSING"
            )


    st.markdown("---")


    # -----------------------------------------------------
    # Four metrics
    # -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        label="📩 Messages Checked",
        value=st.session_state[
            "messages_checked"
        ]
    )


    c2.metric(
        label="🚨 Phishing Detected",
        value=st.session_state[
            "phishing_detected"
        ]
    )


    c3.metric(
        label="👤 Activities Checked",
        value=st.session_state[
            "activities_checked"
        ]
    )


    c4.metric(
        label="⚠️ Suspicious Activities",
        value=st.session_state[
            "suspicious_activities"
        ]
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    st.subheader(
        "Summary"
    )


    last_phish = st.session_state.get(
        "last_phish_results"
    )


    if last_phish is not None:

        total_msgs = len(
            last_phish
        )


        safe_msgs = int(
            (
                last_phish[
                    "label_text"
                ]
                == "🟢 SAFE MESSAGE"
            ).sum()
        )


        phish_msgs = (
            total_msgs
            - safe_msgs
        )


    else:

        total_msgs = (
            st.session_state[
                "messages_checked"
            ]
        )


        safe_msgs = max(
            0,
            st.session_state[
                "messages_checked"
            ]
            -
            st.session_state[
                "phishing_detected"
            ]
        )


        phish_msgs = (
            st.session_state[
                "phishing_detected"
            ]
        )


    last_anom = st.session_state.get(
        "last_anom_results"
    )


    if last_anom is not None:

        total_act = len(
            last_anom
        )


        suspicious_act = int(
            last_anom[
                "is_suspicious"
            ].sum()
        )


        normal_act = (
            total_act
            - suspicious_act
        )


    else:

        total_act = (
            st.session_state[
                "activities_checked"
            ]
        )


        suspicious_act = (
            st.session_state[
                "suspicious_activities"
            ]
        )


        normal_act = max(
            0,
            total_act
            - suspicious_act
        )


    s1, s2, s3, s4 = st.columns(
        4
    )


    s1.markdown(
        f"🟢 Safe Messages\n\n"
        f"**{safe_msgs}**"
    )


    s2.markdown(
        f"🔴 Phishing Messages\n\n"
        f"**{phish_msgs}**"
    )


    s3.markdown(
        f"🟢 Normal Activities\n\n"
        f"**{normal_act}**"
    )


    s4.markdown(
        f"🟠 Suspicious Activities\n\n"
        f"**{suspicious_act}**"
    )


    st.markdown("---")


    st.caption(
        "Metrics reflect activity during the current "
        "Streamlit session. To persist metrics, integrate "
        "a backend or database for logging."
    )