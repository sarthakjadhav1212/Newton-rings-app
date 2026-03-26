import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Newton's Rings Analyzer", layout="centered")

st.title("🔬 Newton's Rings ML Analyzer")

option = st.radio("Select Input Method", ["Manual Input", "Upload CSV", "Upload Excel"])

df = None  # IMPORTANT

# ---- MANUAL INPUT ----
if option == "Manual Input":
    data = st.text_area("Enter data (n,D)", """1,2.1
2,2.9
3,3.6
4,4.2
5,4.8""")

    if data:
        n, D = [], []
        for line in data.strip().split("\n"):
            try:
                a, b = line.split(",")
                n.append(float(a))
                D.append(float(b))
            except:
                st.error("Invalid format")
                st.stop()

        df = pd.DataFrame({"n": n, "D": D})

# ---- CSV ----
elif option == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

# ---- EXCEL ----
elif option == "Upload Excel":
    file = st.file_uploader("Upload Excel", type=["xlsx"])
    if file:
        df = pd.read_excel(file)

if df is not None:
    df["D2"] = df["D"]**2
    st.subheader("📊 Data Preview")
    st.write(df)

    # Compute D²
    

    X = df["n"].values.reshape(-1,1)
    y = df["D2"].values

    # ML model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    m = model.coef_[0]
    b = model.intercept_
    r2 = r2_score(y, y_pred)

    # Manual slope
    n1, n2 = df["n"].iloc[0], df["n"].iloc[-1]
    d1, d2 = df["D2"].iloc[0], df["D2"].iloc[-1]
    m_manual = (d2 - d1) / (n2 - n1)

    # Plot
    st.subheader("📊 D² vs n Plot")
    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, y_pred)
    plt.xlabel("n")
    plt.ylabel("D²")
    st.pyplot(plt)

    # Residuals
    residuals = y - y_pred
    st.subheader("📉 Residual Plot")
    plt.figure()
    plt.scatter(X, residuals)
    plt.axhline(0)
    plt.xlabel("n")
    plt.ylabel("Residuals")
    st.pyplot(plt)

    # Results
    st.subheader("📈 Results")
    st.write(f"ML Graogh Slope: {m:.4f}")
    st.write(f"Manual Slope by Calculation: {m_manual:.4f}")
    st.write(f"Intercept contant : {b:.4f}")
    st.write(f"R² Score acuracy of the model : {r2:.4f}")

    # Physics calculation
    lam = 5.89e-7

    R_manual = m_manual / (4 * lam)
    R_ml = m / (4 * lam)

    st.write(f"Radius of curvature of the len  (manual method): {R_manual:.4f} mm")
    st.write(f"Raiud of the curvature by graph (ML method): {R_ml:.4f} mm")
