import streamlit as st
import pickle
import numpy as np
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =====================================================
# GLOBAL STYLE
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

#. means you're selecting a CSS class.
.main {
    background: radial-gradient(circle at top left, #eef6fb 0%, #e7f0f7 40%, #eef3f8 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2545 0%, #0B6E99 100%);
}
section[data-testid="stSidebar"] * {
    color: #eaf4fa !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 8px 10px;
    margin-bottom: 6px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.16);
}

/* Hero title */
.hero {
    background: linear-gradient(135deg, #0B6E99 0%, #0891B2 55%, #06B6D4 100%);
    color: white;
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 14px 34px rgba(11,110,153,0.30);
    position: relative;
    overflow: hidden;
}
.hero h1 {
    margin-bottom: 6px;
    font-weight: 800;
    font-size: 2.1rem;
}
.hero p {
    opacity: 0.92;
    font-size: 16px;
    margin: 0;
}

/* Content cards */
.box {
    background: white;
    padding: 24px 28px;
    border-radius: 18px;
    box-shadow: 0 6px 22px rgba(15,45,70,0.07);
    margin-top: 16px;
    border: 1px solid #eef2f6;
}

/* Feature cards on home page */
.feature-card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(15,45,70,0.06);
    border: 1px solid #eef2f6;
    height: 100%;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 30px rgba(11,110,153,0.18);
}
.feature-icon {
    font-size: 38px;
    margin-bottom: 10px;
}
.feature-title {
    font-weight: 700;
    font-size: 16.5px;
    color: #0B6E99;
    margin-bottom: 6px;
}
.feature-text {
    font-size: 13.5px;
    color: #5b6b7c;
    line-height: 1.55;
}

/* Step indicator */
.stepper {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}
.step-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: white;
    border: 1px solid #dfe9f0;
    border-radius: 30px;
    padding: 6px 16px 6px 8px;
    font-size: 13px;
    font-weight: 600;
    color: #7c8b9a;
}
.step-pill.active {
    background: linear-gradient(135deg, #0B6E99 0%, #0891B2 100%);
    color: white;
    border: none;
    box-shadow: 0 4px 12px rgba(11,110,153,0.30);
}
.step-num {
    background: rgba(255,255,255,0.25);
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
}
.step-pill:not(.active) .step-num {
    background: #eef2f6;
    color: #7c8b9a;
}
.step-line {
    flex: 1;
    height: 2px;
    background: #dfe9f0;
    max-width: 60px;
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(135deg, #0B6E99 0%, #0891B2 100%);
    color: white;
    font-size: 16px;
    font-weight: 600;
    height: 50px;
    border: none;
    box-shadow: 0 6px 16px rgba(11,110,153,0.28);
    transition: opacity 0.15s ease, transform 0.1s ease;
}
.stButton>button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    color: white;
}

/* Patient info banner */
.patient-banner {
    background: linear-gradient(135deg, #0B6E99 0%, #0891B2 100%);
    color: white;
    padding: 16px 22px;
    border-radius: 14px;
    font-size: 15px;
    font-weight: 500;
    box-shadow: 0 6px 16px rgba(11,110,153,0.22);
}

/* Result panel */
.result-card {
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 26px rgba(15,45,70,0.18);
}
.result-low { background: linear-gradient(135deg, #16A34A 0%, #22C55E 100%); }
.result-high { background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%); }
.result-label {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.result-sub {
    font-size: 14px;
    opacity: 0.92;
    margin-top: 6px;
}

/* Recommendation chip */
.rec-item {
    background: #F0F9FB;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 14px;
    font-size: 14.5px;
    color: #0B2545;
}

hr {
    margin: 1.4em 0;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# NAVIGATION STATE
# =====================================================
# "page" is our own state variable (safe to set anywhere, e.g. from a button).
# The sidebar radio uses a *different* key ("nav_radio") and stays in sync
# with "page" via on_change, which avoids Streamlit's rule that a widget's
# own state key can't be reassigned after that widget has been created.

PAGES = ["Home", "Lifestyle Assessment", "Clinical Prediction", "About"]
PAGE_ICONS = {"Home": "🏠", "Lifestyle Assessment": "📝", "Clinical Prediction": "🩺", "About": "ℹ️"}

if "page" not in st.session_state:
    st.session_state.page = "Home"


def _sync_page_from_sidebar():
    st.session_state.page = st.session_state.nav_radio


st.sidebar.title("🩺 Navigation")

st.sidebar.radio(
    "Go To",
    PAGES,
    index=PAGES.index(st.session_state.page),
    key="nav_radio",
    format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
    on_change=_sync_page_from_sidebar,
    label_visibility="collapsed",
)

page = st.session_state.page

st.sidebar.markdown("---")
st.sidebar.success("Heart Disease Risk Assessment")
st.sidebar.write("**Algorithm:**")
st.sidebar.write("Random Forest")
st.sidebar.write("**Developer:**")
st.sidebar.write("Priti Rajmane")


def goto(page_name: str):
    """Navigate programmatically (e.g. from an in-page button) and rerun.

    Only touches our own 'page' variable. The sidebar radio is created fresh
    on the next run with index=PAGES.index(st.session_state.page), so it
    picks up the new page on its own -- its widget key ('nav_radio') must
    never be written to directly, since Streamlit forbids modifying a
    widget's key after that widget has been instantiated in a run.
    """
    st.session_state.page = page_name
    st.rerun()


def stepper(active_step: int):
    """Render a 2-step progress pill (1 = Lifestyle, 2 = Clinical)."""
    step1_class = "step-pill active" if active_step == 1 else "step-pill"
    step2_class = "step-pill active" if active_step == 2 else "step-pill"
    st.markdown(f"""
    <div class="stepper">
        <div class="{step1_class}"><span class="step-num">1</span> Lifestyle</div>
        <div class="step-line"></div>
        <div class="{step2_class}"><span class="step-num">2</span> Clinical Report</div>
    </div>
    """, unsafe_allow_html=True)


def persist_number(label, key, backup_key, min_value, max_value, default):
    """Number input with a stable widget key (fixes +/- stepper glitches)
    that re-seeds itself from a plain backup key if it was cleared by
    navigating away from this page, and keeps that backup in sync."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(backup_key, default)
    value = st.number_input(label, min_value, max_value,st.session_state[key], key=key)
    st.session_state[backup_key] = value
    return value


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🩺 Heart Disease Risk Assessment System (Adult CAD Screening)</h1>
        <p>AI-Based Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
### Welcome 👋

This application estimates the **probability** of heart disease using
Machine Learning.

The assessment is completed in **two simple steps**:

✔ Lifestyle Assessment &nbsp;&nbsp; ➜ &nbsp;&nbsp; ✔ Clinical Report Prediction

The lifestyle assessment helps understand your health habits, while the
final prediction is made from clinical parameters.
""", unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Lifestyle Assessment</div>
            <div class="feature-text">Answer a few simple questions about your daily health habits.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🩺</div>
            <div class="feature-title">Clinical Prediction</div>
            <div class="feature-text">Enter values from your medical report for AI-based analysis.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Risk Probability</div>
            <div class="feature-text">The model tells you whether heart disease is detected or not detected.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if st.button("🚀 Start Assessment"):
        goto("Lifestyle Assessment")

    st.warning("This application is for educational purposes only and should not replace professional medical advice.")
    st.caption("Developed using Python • Streamlit • Scikit-learn")

# =====================================================
# LIFESTYLE ASSESSMENT
# =====================================================

elif page == "Lifestyle Assessment":

    stepper(1)
    st.title("📝 Lifestyle Assessment")
    st.write("Please answer a few simple questions before continuing.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = persist_number("🎂 Age", "age_widget", "age", 25, 80, 40)

        #index tells selectbox which option to display. gender stores the actual option selected.
        gender_opts = ["Male", "Female"]
        gender = st.selectbox("🚻 Gender", gender_opts,
                               index=gender_opts.index(st.session_state.get("gender", "Male")))
        st.session_state["gender"] = gender

        smoke_opts = ["No", "Yes"]
        smoke = st.selectbox("🚬 Do you smoke?", smoke_opts,
                              index=smoke_opts.index(st.session_state.get("smoke", "No")))
        st.session_state["smoke"] = smoke

        alcohol_opts = ["No", "Yes"]
        alcohol = st.selectbox("🍺 Do you consume alcohol?", alcohol_opts,
                                index=alcohol_opts.index(st.session_state.get("alcohol", "No")))
        st.session_state["alcohol"] = alcohol

    with col2:
        exercise_opts = ["Yes", "No"]
        exercise = st.selectbox("🏃 Do you exercise regularly?", exercise_opts,
                                 index=exercise_opts.index(st.session_state.get("exercise", "Yes")))
        st.session_state["exercise"] = exercise

        diabetes_opts = ["No", "Yes"]
        diabetes = st.selectbox("🍬 Do you have diabetes?", diabetes_opts,
                                 index=diabetes_opts.index(st.session_state.get("diabetes", "No")))
        st.session_state["diabetes"] = diabetes

        bp_opts = ["No", "Yes"]
        bp = st.selectbox("🩸 Do you have high blood pressure?", bp_opts,
                           index=bp_opts.index(st.session_state.get("bp_history", "No")))
        st.session_state["bp_history"] = bp

        family_opts = ["No", "Yes"]
        family = st.selectbox("👨‍👩‍👧 Family history of heart disease?", family_opts,
                               index=family_opts.index(st.session_state.get("family_history", "No")))
        st.session_state["family_history"] = family

    st.divider()

    if st.button("Continue to Clinical Prediction ➜"):
        st.session_state["lifestyle_done"] = True
        goto("Clinical Prediction")

# =====================================================
# CLINICAL PREDICTION PAGE
# =====================================================

elif page == "Clinical Prediction":

    stepper(2)
    st.title("🩺 Clinical Report Prediction")

    if "lifestyle_done" not in st.session_state:
        st.warning("Please complete the Lifestyle Assessment first.")
        st.stop()

    age = st.session_state.get("age", 40)
    gender = st.session_state.get("gender", "Male")

    st.markdown(
        f'<div class="patient-banner">👤 Patient: {gender} &nbsp;|&nbsp; 🎂 Age: {age} years</div>',
        unsafe_allow_html=True
    )

    st.write("")
    st.write("Enter the values from your clinical report.")

    st.divider()

    col1, col2 = st.columns(2)

    # ---------------- LEFT COLUMN ----------------

    with col1:
        cp_opts = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        cp = st.selectbox("Chest Pain Type", cp_opts,
                           index=cp_opts.index(st.session_state.get("cp_saved", cp_opts[0])))
        st.session_state["cp_saved"] = cp

        trestbps = persist_number("Resting Blood Pressure (mm Hg)", "trestbps_widget", "trestbps_saved", 80, 250, 120)

        chol = persist_number("Cholesterol (mg/dl)", "chol_widget", "chol_saved", 100, 600, 200)

        fbs_opts = ["No", "Yes"]
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", fbs_opts,
                            index=fbs_opts.index(st.session_state.get("fbs_saved", "No")))
        st.session_state["fbs_saved"] = fbs

        restecg_opts = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        restecg = st.selectbox("Resting ECG", restecg_opts,
                                index=restecg_opts.index(st.session_state.get("restecg_saved", restecg_opts[0])))
        st.session_state["restecg_saved"] = restecg

        thalach = persist_number("Maximum Heart Rate", "thalach_widget", "thalach_saved", 60, 220, 150)

    # ---------------- RIGHT COLUMN ----------------

    with col2:
        exang_opts = ["No", "Yes"]
        exang = st.selectbox("Exercise Induced Angina", exang_opts,
                              index=exang_opts.index(st.session_state.get("exang_saved", "No")))
        st.session_state["exang_saved"] = exang

        oldpeak = persist_number("ST Depression", "oldpeak_widget", "oldpeak_saved", 0.0, 10.0, 1.0)

        slope_opts = ["Upsloping", "Flat", "Downsloping"]
        slope = st.selectbox("ST Segment Slope", slope_opts,
                              index=slope_opts.index(st.session_state.get("slope_saved", slope_opts[0])))
        st.session_state["slope_saved"] = slope

        ca_opts = [0, 1, 2, 3, 4]
        ca = st.selectbox("Major Vessels", ca_opts,
                           index=ca_opts.index(st.session_state.get("ca_saved", 0)))
        st.session_state["ca_saved"] = ca

        thal_opts = ["Normal", "Fixed Defect", "Reversible Defect"]
        thal = st.selectbox("Thalassemia", thal_opts,
                             index=thal_opts.index(st.session_state.get("thal_saved", thal_opts[0])))
        st.session_state["thal_saved"] = thal

    st.divider()

    predict = st.button("🔍 Predict Risk", use_container_width=True)

    if predict:

        sex = 1 if gender == "Male" else 0
        cp_val = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
        fbs_val = 1 if fbs=="Yes" else 0
        restecg_val = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg] #"use the value stored in the variable restecg as the key to look up in this dictionary."
        exang_val = 1 if exang=="Yes" else 0
        slope_val = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
        thal_val = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

        input_data = pd.DataFrame([[
            age, sex, cp_val, trestbps, chol, fbs_val, restecg_val,
            thalach, exang_val, oldpeak, slope_val, ca, thal_val
        ]], columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                      "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

        input_data = scaler.transform(input_data)

        probability = model.predict_proba(input_data)
        risk = probability[0][0]

        if risk < 0.50:
            css_class, label, sub = "result-low", "🟢 Lower Estimated Risk", "Please continue regular health checkups."
        else:
            css_class, label, sub = "result-high", "🔴 Higher Estimated Risk", "Please consult a healthcare professional for further evaluation."

        recs = []
        if age >= 60:
            recs.append("📅 Consider regular cardiovascular health check-ups to monitor heart health and identify potential concerns early.")
        if trestbps > 130:
            recs.append("🩸 Maintain healthy BP through a balanced diet, regular physical activity, stress management and routine monitoring.")
        if chol > 200:
            recs.append("🥗 Choose more fibre-rich foods such as fruits, vegetables, whole grains while limiting foods rich in saturated and trans fats.")
        if fbs_val == 1:
            recs.append("🍬 Maintain consistent blood-sugar management through balanced meals, regular activity and appropriate medical follow-up.")
        if exang_val == 1:
            recs.append("🏃 Discuss exercise intensity with a healthcare professional and stop activity if you experience concerning symptoms such as chest discomfort.")
        if oldpeak > 2:
            recs.append("❤️ Consider reviewing the exercise ECG findings with a healthcare professional for further cardiac assessment.")
        if cp_val in [0, 1]:
            recs.append("⚠️ Avoid ignoring recurring chest discomfort, especially when it occurs during physical activity.")


        if st.session_state.get("smoke") == "Yes":
            recs.append("🚭 Quitting is one of the single most effective steps you can take to reduce cardiovascular risk.")
        if st.session_state.get("alcohol") == "Yes":
            recs.append("🍺 Consider limiting alcohol intake, as excessive consumption can contribute to high blood pressure and heart strain over time.")
        if st.session_state.get("exercise") == "No":
            recs.append("🏃 You indicated limited regular exercise — even light daily activity like walking can meaningfully improve cardiovascular health over time.")
        if st.session_state.get("diabetes") == "Yes":
            recs.append("🩺 Given your reported diabetes, consistent blood sugar monitoring alongside cardiovascular check-ups is especially important.")
        if st.session_state.get("bp_history") == "Yes":
            recs.append("🩸 Since you have a history of high blood pressure, regular home monitoring and medication adherence (if prescribed) are important.")
        if st.session_state.get("family_history") == "Yes":
            recs.append("👨‍👩‍👧 With a family history of heart disease, earlier and more frequent cardiac screening is generally advisable, even without symptoms.")

        recs += [
            "🥦 Prefer a heart-healthy diet rich in vegetables, fruits, whole grains and unsaturated fats.",
            "😴 Maintain a consistent sleep schedule and aim for adequate, good-quality sleep.",
            "💧 Maintain adequate hydration throughout the day, especially during hot weather and physical activity.",
        ]

        # Freeze this exact result in session_state -- editing inputs afterward
        # will NOT change what's displayed until Predict Risk is clicked again.
        st.session_state["result"] = {
            "css_class": css_class, "label": label, "sub": sub, "recs": recs
        }

    if "result" in st.session_state:

        result = st.session_state["result"]

        st.divider()
        st.subheader("📊 Prediction Result")

        st.markdown(f"""
        <div class="result-card {result['css_class']}">
            <div class="result-label">{result['label']}</div>
            <div class="result-sub">{result['sub']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.info("This is an AI-based prediction and should not replace professional medical advice.")

        st.divider()
        st.subheader("💡 Personalized Health Recommendations")

        rec_col1, rec_col2 = st.columns(2)
        for i, rec in enumerate(result["recs"]):
            target = rec_col1 if i % 2 == 0 else rec_col2
            target.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("ℹ️ Go to About"):
            goto("About")

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.markdown("""
    <div class="hero">
        <h1>About the Project</h1>
        <p>AI-Based Heart Disease Risk Assessment System</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
This application is an **Artificial Intelligence (AI) powered Clinical Decision
Support System (CDSS)** developed to detect the **presence of heart disease**
using Machine Learning techniques.

The system analyzes important clinical parameters and provides an **early risk
assessment** to support healthcare awareness. The generated prediction is intended
to assist users in understanding their potential risk and encouraging timely
medical consultation.

### Project Objectives

- Provide an easy-to-use and user-friendly interface
- Estimate heart disease risk using Machine Learning
- Present a clear result rather than a direct medical diagnosis
- Promote preventive healthcare and early awareness
- Demonstrate the practical application of AI in the healthcare sector

### Machine Learning Model

- Algorithm: Random Forest Classifier
- Programming Language: Python
- Framework: Streamlit
- Libraries: Scikit-learn, NumPy, Pandas

### Clinical Parameters Used

Age, Gender, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood
Sugar, Resting ECG, Maximum Heart Rate, Exercise-Induced Angina, ST Depression
(Oldpeak), ST Segment Slope, Number of Major Blood Vessels, Thalassemia

### Key Features

- Interactive and user-friendly interface
- Lifestyle Assessment module
- AI-based risk prediction
- Clear , easy-to-understand result display
- Personalized health recommendations
- Clinical decision support

### Disclaimer

This application is developed **for educational and research purposes only**.
The prediction generated by this system **does not replace professional medical
diagnosis, laboratory investigations, or clinical judgment**. Users are strongly
advised to consult a qualified healthcare professional for accurate diagnosis and
appropriate treatment.
""", unsafe_allow_html=True)

    st.divider()
    st.caption("Developed by Priti Rajmane | Diploma Project | AI-Based Heart Disease Risk Assessment System")
