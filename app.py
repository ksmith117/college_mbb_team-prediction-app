import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="College Basketball Prediction App", layout="centered")

st.title("College Basketball Team Prediction App")

st.markdown("""
### About This

This tool models relationships between team performance and postseason outcomes in men's college basketball.

### Key Outputs
- Postseason Qualification
- Postseason Qualification Probability
- Conference Rank
- Postseason Efficiency
- Efficiency Tier

### Notes
- Predictions are based on historical patterns, not guarantees
- Reverse predictions are approximate
- Postseason efficiency tiers are based on the observed distribution in the dataset
""")

# -----------------------
# LOAD DATA
# -----------------------
data = pd.read_csv("MB_All_Conf.csv")

# Clean column names hard
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w_]", "", regex=True)
)

# Rename known variants to the names used in the app
data = data.rename(columns={
    "conference_ranking": "conf_rank",
    "conference_rank": "conf_rank",
    "postseasoneff": "postseason_eff",
    "postseason_efficiency": "postseason_eff",
    "conf_win_percent": "conf_win_pct",
    "conf_win_percentage": "conf_win_pct",
    "win_percent": "win_pct",
    "ncaa_tournament": "postseason"
})

# Fix availability if stored like percentages
if data["availability"].dtype == "object":
    data["availability"] = (
        data["availability"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float) / 100
    )

# Drop missing values needed for modeling
data = data.dropna(subset=[
    "conference",
    "availability",
    "conf_win_pct",
    "postseason",
    "conf_rank",
    "postseason_eff"
])

# Make sure postseason is numeric
data["postseason"] = data["postseason"].astype(int)

# -----------------------
# CONFERENCE FILTER
# -----------------------
conferences = sorted(data["conference"].dropna().unique())
selected_conf = st.selectbox("Select Conference", conferences)

filtered_data = data[data["conference"] == selected_conf]

if len(filtered_data) < 5:
    st.warning("Not enough data for this conference.")
    st.stop()

# -----------------------
# EFFICIENCY TIER FUNCTION
# Basketball-specific thresholds
# -----------------------
def get_tier(eff):
    if eff <= 0:
        return "No Postseason Appearance"
    elif eff < 1.0:
        return "Below Average"
    elif eff < 1.75:
        return "Average"
    elif eff < 2.75:
        return "Strong"
    else:
        return "Elite"

# -----------------------
# TRAIN MODELS
# -----------------------

# Forward models
X1 = filtered_data[["availability", "conf_win_pct"]]
y_post = filtered_data["postseason"]
y_rank = filtered_data["conf_rank"]
y_eff = filtered_data["postseason_eff"]

model_post = RandomForestClassifier(random_state=42)
model_rank = RandomForestRegressor(random_state=42)
model_eff = RandomForestRegressor(random_state=42)

model_post.fit(X1, y_post)
model_rank.fit(X1, y_rank)
model_eff.fit(X1, y_eff)

# Reverse models
X2 = filtered_data[["postseason", "conf_rank", "postseason_eff"]]
y_avail = filtered_data["availability"]
y_conf = filtered_data["conf_win_pct"]

model_avail = RandomForestRegressor(random_state=42)
model_conf = RandomForestRegressor(random_state=42)

model_avail.fit(X2, y_avail)
model_conf.fit(X2, y_conf)

# -----------------------
# MODE SWITCH
# -----------------------
mode = st.radio("Choose Prediction Mode", ["Forward", "Reverse"])

# -----------------------
# FORWARD MODE
# -----------------------
if mode == "Forward":
    st.subheader(f"Forward Prediction — {selected_conf}")

    availability = st.number_input(
        "Availability",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01
    )

    conf_win_pct = st.number_input(
        "Conference Win %",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01
    )

    if st.button("Predict Forward"):
        X = pd.DataFrame([{
            "availability": availability,
            "conf_win_pct": conf_win_pct
        }])

        post = int(model_post.predict(X)[0])
        prob = float(model_post.predict_proba(X)[0][1])
        rank = float(model_rank.predict(X)[0])
        eff = float(model_eff.predict(X)[0])

        if eff < 0:
            eff = 0.0

        tier = get_tier(eff)

        st.write("### Results")
        st.write(f"**Conference:** {selected_conf}")
        st.write(f"**Postseason Qualification:** {post}")
        st.write(f"**Postseason Qualification Probability:** {round(prob, 3)}")
        st.write(f"**Conference Rank:** {round(rank, 2)}")
        st.write(f"**Postseason Efficiency:** {round(eff, 3)}")
        st.write(f"**Efficiency Tier:** {tier}")

# -----------------------
# REVERSE MODE
# -----------------------
else:
    st.subheader(f"Reverse Prediction — {selected_conf}")

    postseason = st.number_input(
        "Postseason (0 or 1)",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )

    conf_rank = st.number_input(
        "Conference Rank",
        min_value=1.0,
        max_value=25.0,
        value=5.0,
        step=0.1
    )

    postseason_eff = st.number_input(
        "Postseason Efficiency",
        min_value=0.0,
        max_value=5.0,
        value=1.50,
        step=0.01
    )

    if st.button("Predict Reverse"):
        X = pd.DataFrame([{
            "postseason": postseason,
            "conf_rank": conf_rank,
            "postseason_eff": postseason_eff
        }])

        avail_pred = float(model_avail.predict(X)[0])
        conf_pred = float(model_conf.predict(X)[0])

        if avail_pred < 0:
            avail_pred = 0.0
        elif avail_pred > 1:
            avail_pred = 1.0

        if conf_pred < 0:
            conf_pred = 0.0
        elif conf_pred > 1:
            conf_pred = 1.0

        tier = get_tier(postseason_eff)

        st.write("### Results")
        st.write(f"**Conference:** {selected_conf}")
        st.write(f"**Predicted Availability:** {round(avail_pred, 3)}")
        st.write(f"**Predicted Conference Win %:** {round(conf_pred, 3)}")
        st.write(f"**Efficiency Tier:** {tier}")

# -----------------------
# INTERPRETATION
# -----------------------
st.markdown("""
---
### Efficiency Interpretation (Basketball)

- 0.00 → No Postseason Appearance
- 0.01–0.99 → Below Average
- 1.00–1.74 → Average
- 1.75–2.74 → Strong
- 2.75+ → Elite

These ranges are based on the observed postseason efficiency distribution in the dataset.
""")
