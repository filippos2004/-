import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 🔧 Function to embed local background image
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
        }}
        .title {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
        }}
        .section {{
            background-color: rgba(0, 0, 0, 0.5);
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# 🌌 Apply background
set_bg_from_local("sky.jpg")

# 🎯 Main title
st.markdown("<div class='title'>🔬 Molebio Data Analysis</div>", unsafe_allow_html=True)

# 📁 Upload section
st.markdown("<div class='section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Ανεβάστε το dataset σας (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("#### 🔍 Προεπισκόπηση Δεδομένων:")
    st.dataframe(df.head())

    # 📊 Στατιστική ανάλυση
    if st.checkbox("📈 Εμφάνιση περιγραφικής στατιστικής"):
        st.write(df.describe())

    # ✏️ Επιλογές χαρακτηριστικών & στόχου
    features = st.multiselect("🧬 Επιλέξτε χαρακτηριστικά (features)", df.columns.tolist())
    target = st.selectbox("🎯 Επιλέξτε την ετικέτα (target)", df.columns.tolist())

    if features and target:
        X = df[features]
        y = df[target]

        # 🔀 Διαχωρισμός dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 🌲 Εκπαίδευση μοντέλου
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        # ✅ Αξιολόγηση
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Ακρίβεια μοντέλου: {accuracy:.2f}")

    # 📊 Οπτικοποίηση
    st.markdown("### 📊 Οπτικοποίηση Δεδομένων")
    column = st.selectbox("📌 Επιλέξτε μεταβλητή για ανάλυση", df.columns)

    # Αν η στήλη είναι κατηγορική (object ή category), κάνουμε οριζόντιο bar plot
    if df[column].dtype == 'object' or str(df[column].dtype).startswith('category'):
        fig, ax = plt.subplots(figsize=(8, max(6, len(df[column].unique()) * 0.3)))
        order = df[column].value_counts().index
        sns.countplot(y=df[column], order=order, ax=ax)
        ax.set_ylabel(column)
        ax.set_xlabel('Count')
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        # Numeric data: histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[column], kde=True, ax=ax)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        fig.subplots_adjust(bottom=0.25)
        st.pyplot(fig)

    # 🔍 Scatter plot για δύο χαρακτηριστικά
    if len(features) >= 2:
        st.markdown("### 🔍 Scatter Plot")
        x_axis = st.selectbox("📉 Επιλέξτε X άξονα", features)
        y_axis = st.selectbox("📈 Επιλέξτε Y άξονα", features)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[target], ax=ax2)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        fig2.subplots_adjust(bottom=0.25)
        st.pyplot(fig2)

st.markdown("</div>", unsafe_allow_html=True)
