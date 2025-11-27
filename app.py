import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODELS & TRANSFORMER
# ==============================
rf = joblib.load("Models/final_rf.pkl")
iforest = joblib.load("Models/iforest.pkl")
qt = joblib.load("Models/quantile_transformer.pkl")

# Features each component expects
qt_features = list(qt.feature_names_in_)              # features used when QT was fit
rf_features = list(rf.feature_names_in_)              # features RF was trained on
if hasattr(iforest, "feature_names_in_"):
    iso_features = list(iforest.feature_names_in_)    # features IsolationForest was trained on
else:
    iso_features = rf_features

# A single "raw-like" baseline row for QT: all zeros
baseline_raw = pd.DataFrame([0.0] * len(qt_features), index=qt_features).T


# ==============================
# BUILD SCALED INPUT ROW
# ==============================
def build_scaled_rows(amount, card_group, addr, c1, v200):
    """
    1) Start from a raw-like row with all qt_features.
    2) Overwrite a few important raw values based on the UI.
    3) Pass through QuantileTransformer.
    4) Extract exactly the columns RF and IsolationForest expect.
    """

    raw = baseline_raw.copy()

    # Map UI values into raw feature space
    if "TransactionAmt" in raw.columns:
        raw.loc[0, "TransactionAmt"] = float(amount)

    if "card1" in raw.columns:
        raw.loc[0, "card1"] = float(card_group)

    if "addr1" in raw.columns:
        raw.loc[0, "addr1"] = float(addr)

    if "C1" in raw.columns:
        raw.loc[0, "C1"] = float(c1)

    if "V200" in raw.columns:
        raw.loc[0, "V200"] = float(v200)

    # Ensure columns are in the exact order QT expects
    raw = raw[qt_features]

    # Transform using the fitted QuantileTransformer
    scaled_full = qt.transform(raw)
    scaled_df = pd.DataFrame(scaled_full, columns=qt_features)

    # Slice down to exactly the features RF and IForest saw during training
    X_rf = scaled_df[[f for f in rf_features if f in scaled_df.columns]]
    X_iso = scaled_df[[f for f in iso_features if f in scaled_df.columns]]

    return X_rf, X_iso


# ==============================
# MODEL PREDICTION (BASE)
# ==============================
def base_model_predict(amount, card_group, addr, device, c1, v200):
    """
    Get the raw RF fraud probability and anomaly score from IsolationForest.
    Device is only used for heuristic boosting, not for the models directly.
    """
    X_rf, X_iso = build_scaled_rows(amount, card_group, addr, c1, v200)

    # RF probability of fraud
    prob = float(rf.predict_proba(X_rf)[0][1])

    # IsolationForest anomaly score: higher = more anomalous (flip sign)
    try:
        anom = -float(iforest.score_samples(X_iso)[0])
    except Exception:
        anom = 0.0

    return prob, anom


# ==============================
# HEURISTIC BOOSTING (FOR DEMO)
# ==============================
def boosted_probability(base_prob, amount, card_group, addr, device, c1, v200, is_fraud_preset=False):
    """
    Boost fraud probability in clearly suspicious cases so you can demo both
    legitimate and fraudulent examples easily. Keeps low-risk realistic, but
    pushes extreme cases above 50–90%.
    """
    boost = 0.0

    # Very large amounts → suspicious
    if amount > 5000:
        boost += 0.20
    if amount > 10000:
        boost += 0.20
    if amount > 20000:
        boost += 0.25

    # Weird card group (last 4 digits far from typical mid-range)
    if card_group < 500 or card_group > 18000:
        boost += 0.15

    # Rare / extreme address codes
    if addr in [0, 1, 500]:
        boost += 0.10

    # Desktop devices more suspicious for high-amount transactions
    if device in ["Windows", "MacOS"] and amount > 3000:
        boost += 0.15

    # Abnormal behavior score
    if c1 > 5:
        boost += 0.20
    if c1 > 8:
        boost += 0.25

    # Strong negative security score → very suspicious
    if v200 < -1.0:
        boost += 0.20
    if v200 < -2.0:
        boost += 0.25

    # Fraud preset: make sure it's clearly fraud for demo
    if is_fraud_preset:
        boost = max(boost, 0.45)  # at least +45% extra

    final = base_prob + boost
    return min(final, 0.999)


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="FraudGen Demo", layout="centered")

st.title("💳 FraudGen — Transaction Risk Demo")
st.write(
    "This demo combines a trained Random Forest model, anomaly detection, "
    "and heuristic rules to flag **online payment fraud**."
)

# -------- Preset selector --------
preset = st.radio(
    "Choose a scenario:",
    ["🟢 Legitimate Example", "🔴 Fraud Example", "✍ Custom Input"],
    horizontal=True
)

# -------- Default values based on preset --------
if preset == "🟢 Legitimate Example":
    amount = 95.0
    card_group = 5400
    addr = 120
    device = "Android"
    c1 = 1.0
    v200 = 0.5
    is_fraud_preset = False

elif preset == "🔴 Fraud Example":
    amount = 22000.0
    card_group = 999
    addr = 0
    device = "Windows"
    c1 = 9.0
    v200 = -2.5
    is_fraud_preset = True

else:
    is_fraud_preset = False
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("💰 Transaction Amount (USD)", min_value=1.0, max_value=50000.0, value=120.0, step=10.0)
        card_group = st.number_input("💳 Last 4 digits of card", min_value=100, max_value=20000, value=4600, step=100)
        addr = st.number_input("📍 Billing Address Code (addr1)", min_value=0, max_value=500, value=120, step=5)

    with col2:
        device = st.selectbox("📱 Device Type", ["Android", "iOS", "Windows", "MacOS"])
        c1 = st.slider("🧠 C1 — Behavior Score", min_value=-1.0, max_value=10.0, value=1.0, step=0.5)
        v200 = st.slider("🔐 V200 — Security Signal", min_value=-3.0, max_value=3.0, value=0.2, step=0.1)

st.markdown("---")

if st.button("🚦 Evaluate Transaction Risk"):
    # 1) Get raw model outputs
    base_prob, anomaly = base_model_predict(amount, card_group, addr, device, c1, v200)

    # 2) Apply heuristic boosting (for demo visibility)
    final_prob = boosted_probability(
        base_prob,
        amount,
        card_group,
        addr,
        device,
        c1,
        v200,
        is_fraud_preset=is_fraud_preset
    )

    st.subheader("📊 Risk Assessment")

    st.write(f"**Base Model Fraud Probability:** `{base_prob*100:.2f}%`")
    st.write(f"**Boosted Demo Fraud Probability:** `{final_prob*100:.2f}%`")
    st.write(f"**Anomaly Score:** `{anomaly:.3f}`  (higher ≈ more unusual)")

    st.markdown("---")

    if final_prob >= 0.55:
        st.error("🚨 **HIGH FRAUD RISK — Block or Review Manually**")
        st.write(
            "- Unusually high amount\n"
            "- Suspicious card/address pattern\n"
            "- Behavior and security signals indicate potential fraud"
        )
    else:
        st.success("🟢 **LOW RISK — Transaction Appears Legitimate**")
        st.write(
            "- Amount within normal range\n"
            "- Card and address look typical\n"
            "- No strong fraud signals detected"
        )

st.caption("⚙ Powered by Random Forest + Isolation Forest + synthetic fraud heuristics (demo-boosted).")

