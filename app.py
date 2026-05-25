import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------------
# LOAD CSS
# ------------------------------------------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv("titanic.csv")

# ------------------------------------------------
# DATA CLEANING
# ------------------------------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# LABEL ENCODING

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# FAMILY SIZE

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# FEATURES

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize"
]

X = df[features]
y = df["Survived"]

# ------------------------------------------------
# SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------
# MODEL
# ------------------------------------------------

@st.cache_resource
def train_model():

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

model = train_model()

# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown("""
<div class="main-title">
🚢 Titanic Survival Prediction
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Machine Learning Based Smart Passenger Survival Analysis
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TOP METRICS
# ------------------------------------------------

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

with m2:
    st.metric("Dataset Size", len(df))

with m3:
    st.metric("Testing Samples", len(y_test))

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------

left, right = st.columns([1.1, 1])

# ------------------------------------------------
# LEFT SIDE
# ------------------------------------------------

with left:

    st.markdown("""
    <div class="card">
    <h2>Passenger Information</h2>
    </div>
    """, unsafe_allow_html=True)

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    fare = st.slider(
        "Fare",
        0,
        600,
        50
    )

    embarked = st.selectbox(
        "Embarked Port",
        ["S", "C", "Q"]
    )

    family = st.slider(
        "Family Size",
        1,
        10,
        1
    )

    predict = st.button("Predict Survival")

# ------------------------------------------------
# RIGHT SIDE
# ------------------------------------------------

with right:

    st.markdown("""
    <div class="card">
    <h2>Confusion Matrix</h2>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.imshow(cm)

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16
            )

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if predict:

    sex_val = 1 if sex == "Male" else 0

    embarked_map = {
        "S": 2,
        "C": 0,
        "Q": 1
    }

    embarked_val = embarked_map[embarked]

    user_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_val,
        "Age": age,
        "Fare": fare,
        "Embarked": embarked_val,
        "FamilySize": family
    }])

    prediction = model.predict(user_data)[0]

    probability = model.predict_proba(user_data)[0]

    survive_prob = probability[1] * 100
    nonsurvive_prob = probability[0] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:

        st.success(
            f"Passenger is likely to SURVIVE ({survive_prob:.2f}%)"
        )

    else:

        st.error(
            f"Passenger is likely to NOT SURVIVE ({nonsurvive_prob:.2f}%)"
        )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Survival Chance", f"{survive_prob:.2f}%")

    with r2:
        st.metric("Non Survival", f"{nonsurvive_prob:.2f}%")

    with r3:
        st.metric(
            "Prediction",
            "Survived" if prediction == 1 else "Not Survived"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4,4))

    ax2.pie(
        [survive_prob, nonsurvive_prob],
        labels=["Survive", "Not Survive"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig2)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)