import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# --------------------------------
# LOAD CSS
# --------------------------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# --------------------------------
# LOAD DATASET
# --------------------------------

df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(
    df["Age"].mean()
)

features = [
    "Pclass",
    "Age",
    "Fare"
]

X = df[features]
y = df["Survived"]

# --------------------------------
# NORMALIZATION
# --------------------------------

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# --------------------------------
# BUILD ANN MODEL
# --------------------------------

@st.cache_resource
def train_model():

    model = MLPClassifier(
        hidden_layer_sizes=(8,4),
        activation='relu',
        max_iter=500,
        random_state=42
    )

    model.fit(X,y)

    return model

model = train_model()

# --------------------------------
# HEADER
# --------------------------------

st.markdown(
"""
<div class='title'>
🚢 Titanic Survival Prediction System
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Deep Learning Based Passenger Survival Prediction
</div>
""",
unsafe_allow_html=True
)

c1,c2,c3 = st.columns([2,1,2])

with c2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2784/2784445.png",
        width=100
    )

# --------------------------------
# DESCRIPTION
# --------------------------------

st.markdown(
"""
<div class='card'>

<h3>Project Description</h3>

<p>
This application predicts whether a passenger
would survive using an Artificial Neural Network.
</p>

<p>Technologies Used:</p>

<p>✅ ANN Model</p>
<p>✅ Streamlit</p>
<p>✅ Titanic Dataset</p>
<p>✅ Machine Learning Deployment</p>

</div>
""",
unsafe_allow_html=True
)

# --------------------------------
# INPUT SECTION
# --------------------------------

st.markdown(
"""
<div class='card'>
<h3>Passenger Input Form</h3>
</div>
""",
unsafe_allow_html=True
)

col1,col2,col3 = st.columns(3)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

with col2:

    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

# --------------------------------
# PREDICT BUTTON
# --------------------------------

if st.button(
    "Predict Survival"
):

    user = np.array(
        [[pclass,age,fare]]
    )

    user = scaler.transform(user)

    prob = model.predict_proba(
        user
    )[0][1]

    non_prob = 1 - prob

    if prob > 0.5:

        result = "Survived"

        st.success(
            "Passenger likely survives"
        )

    else:

        result = "Not Survived"

        st.error(
            "Passenger likely may not survive"
        )

    confidence = max(
        prob,
        non_prob
    )*100

# --------------------------------
# OUTPUT
# --------------------------------

    st.markdown(
    """
    <div class='card'>
    <h3>Prediction Output</h3>
    </div>
    """,
    unsafe_allow_html=True
    )

    a,b,c = st.columns(3)

    with a:

        st.metric(
            "Prediction",
            result
        )

    with b:

        st.metric(
            "Survival Probability",
            f"{prob*100:.2f}%"
        )

    with c:

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

# --------------------------------
# SMALL GRAPH
# --------------------------------

    st.markdown(
    """
    <div class='card'>
    <h3>Probability Visualization</h3>
    </div>
    """,
    unsafe_allow_html=True
    )

    x,y,z = st.columns([1,2,1])

    with y:

        fig,ax = plt.subplots(
            figsize=(3,3)
        )

        labels = [
            "Survival",
            "Non Survival"
        ]

        values = [
            prob,
            non_prob
        ]

        explode = [0.05,0]

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            explode=explode
        )

        st.pyplot(
            fig,
            use_container_width=False
        )

# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.caption(
    "Built using Streamlit + ANN"
)