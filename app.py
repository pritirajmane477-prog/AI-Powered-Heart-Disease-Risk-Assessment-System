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
.result-mid { background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%); }
.result-high { background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%); }
.result-pct {
    font-size: 46px;
    font-weight: 800;
    margin: 6px 0 2px 0;
}
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
    background: white;
    border: 1px solid #eef2f6;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14.5px;
    color: #37454f;
    box-shadow: 0 2px 8px rgba(15,45,70,0.04);
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
            <div class="feature-text">The model predicts heart disease based on assessment.</div>
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
        age = st.number_input("🎂 Age", min_value=25, max_value=80, value=40)
        gender = st.selectbox("🚻 Gender", ["Male", "Female"])
        smoke = st.selectbox("🚬 Do you smoke?", ["No", "Yes"])
        alcohol = st.selectbox("🍺 Do you consume alcohol?", ["No", "Yes"])

    with col2:
        exercise = st.selectbox("🏃 Do you exercise regularly?", ["Yes", "No"])
        diabetes = st.selectbox("🍬 Do you have diabetes?", ["No", "Yes"])
        bp = st.selectbox("🩸 Do you have high blood pressure?", ["No", "Yes"])
        family = st.selectbox("👨‍👩‍👧 Family history of heart disease?", ["No", "Yes"])

    st.divider()

    if st.button("Continue to Clinical Prediction ➜"):

        # Persist lifestyle inputs for later pages / recommendations
        st.session_state["age"] = age
        st.session_state["gender"] = gender
        st.session_state["smoke"] = smoke
        st.session_state["alcohol"] = alcohol
        st.session_state["exercise"] = exercise
        st.session_state["diabetes"] = diabetes
        st.session_state["bp_history"] = bp
        st.session_state["family_history"] = family

        goto("Clinical Prediction")

# =====================================================
# CLINICAL PREDICTION PAGE
# =====================================================

elif page == "Clinical Prediction":

    stepper(2)
    st.title("🩺 Clinical Report Prediction")

    age = st.session_state.get("age", 45)
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
        cp = st.selectbox(
         "Chest Pain Type",
           [0, 1, 2, 3]
         )

        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)

        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

        fbs = st.selectbox(
         "Fasting Blood Sugar >120 mg/dl", ["No","Yes"] )

        restecg = st.selectbox(
         "Resting ECG",
         [0, 1, 2]
        )

        thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)


        
    # ---------------- RIGHT COLUMN ----------------

    with col2:
        exang = st.selectbox( "Exercise Induced Angina",  ["No","Yes"]  )
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = st.selectbox( "ST Segment Slope", [0, 1, 2] )
        ca = st.selectbox( "Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [1, 2, 3])




        
    st.divider()

    predict = st.button("🔍 Predict Risk", use_container_width=True)

    if predict:

        sex = 1 if gender == "Male" else 0
        cp_val = cp
        fbs_val = 1 if fbs=="Yes" else 0
        restecg_val = restecg
        exang_val = 1 if exang=="Yes" else 0
        slope_val = slope
        thal_val = thal

        input_data = pd.DataFrame([[
            age, sex, cp_val, trestbps, chol, fbs_val, restecg_val,
            thalach, exang_val, oldpeak, slope_val, ca, thal_val
        ]], columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                      "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

        input_data = scaler.transform(input_data)

        probability = model.predict_proba(input_data)
        # Note: in this dataset, target=1 actually means "no disease" and
        # target=0 means "disease present" (confirmed against the data's own
        # clinical averages) -- so P(disease) is probability[0][0], not [1].
        risk = probability[0][0]

        st.divider()
        st.subheader("📊 Prediction Result")

        if risk < 0.50:
            css_class, label, sub = "result-low", "🟢 Heart Disease Not Detected", "Please continue regular health checkups."
        else:
            css_class, label, sub = "result-high", "🔴 Heart Disease Detected", "Please consult a cardiologist for further evaluation."
        
        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-label">{label}</div>
            <div class="result-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.info("This is an AI-based prediction and should not replace professional medical advice.")

        st.divider()
        st.subheader("💡 Personalized Health Recommendations")

        recs = []
        if age >= 60:
            recs.append("📅 Schedule regular heart check-ups.")
        if trestbps > 130:
            recs.append("🩸 Monitor your blood pressure regularly.")
        if chol > 200:
            recs.append("🥗 Reduce foods high in cholesterol and saturated fats.")
        if fbs_val == 1:
            recs.append("🍬 Keep your blood sugar under control.")
        if exang_val == 1:
            recs.append("🏃 Avoid strenuous exercise without medical advice.")
        if oldpeak > 2:
            recs.append("❤️ Visit a cardiologist for further evaluation.")
        if cp_val in [2, 3]:
            recs.append("⚠️ Do not ignore chest pain symptoms.")

        recs += [
            "🥦 Eat more fruits and vegetables.",
            "🚶 Exercise regularly.",
            "😴 Sleep 7–8 hours every day.",
            "🚭 Avoid smoking and tobacco.",
            "💧 Drink enough water.",
        ]

        rec_col1, rec_col2 = st.columns(2)
        for i, rec in enumerate(recs):
            target = rec_col1 if i % 2 == 0 else rec_col2
            target.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

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
Support System (CDSS)** developed to estimate the **probability of heart disease**
using Machine Learning techniques.

The system analyzes important clinical parameters and provides an **early risk
assessment** to support healthcare awareness. The generated prediction is intended
to assist users in understanding their potential risk and encouraging timely
medical consultation.

### Project Objectives

- Provide an easy-to-use and user-friendly interface
- Estimate heart disease risk using Machine Learning
- Present prediction results as a probability rather than a direct diagnosis
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
- Probability-based result visualization
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
