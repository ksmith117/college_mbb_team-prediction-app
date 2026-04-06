import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="College Basketball Prediction App", layout="centered")


def main():
    st.title("College Basketball Team Prediction App")

    st.markdown("""
    ### About This

    This tool models relationships between team performance and postseason outcomes in men's college basketball.

    ### Key Outputs
    - Conference Tournament Qualification
    - NCAA Tournament Qualification
    - NCAA Tournament Qualification Probability
    - Conference Rank
    - Postseason Efficiency
    - Efficiency Tier

    ### Notes
    - Predictions are based on historical patterns, not guarantees
    - Reverse predictions are approximate
    - Source data was compiled from official athletics websites for the 2025–2026 men's college basketball season
    - Postseason efficiency in this model is based on conference tournament performance
    - NCAA tournament qualification is modeled separately
    - Postseason efficiency tiers are based on the observed distribution in the dataset
    """)

    # -----------------------
    # HELPERS
    # -----------------------
    def clean_numeric(series, percent=False):
        s = (
            series.astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
        s = pd.to_numeric(s, errors="coerce")

        if percent and len(s.dropna()) > 0 and s.dropna().max() > 1:
            s = s / 100

        return s

    def get_tier(eff):
        if eff <= 0:
            return "No Conference Tournament Appearance"
        elif eff < 1.0:
            return "Below Average"
        elif eff < 1.5:
            return "Average"
        elif eff < 3.0:
            return "Strong"
        else:
            return "Elite"

    def get_forward_tier_text(tier):
        if tier == "Below Average":
            return "This efficiency level suggests a weaker conference tournament profile."
        elif tier == "Average":
            return "This efficiency level suggests a moderate conference tournament profile."
        elif tier == "Strong":
            return "This efficiency level suggests a stronger conference tournament profile and may imply a more competitive overall postseason resume."
        elif tier == "Elite":
            return "This efficiency level suggests a very strong conference tournament profile and may imply one of the strongest overall postseason resumes."
        else:
            return "This team does not currently project as having a meaningful conference tournament profile."

    def get_reverse_tier_text(tier):
        if tier == "Below Average":
            return "that efficiency tier would reflect a weaker conference tournament profile."
        elif tier == "Average":
            return "that efficiency tier would reflect a moderate conference tournament profile."
        elif tier == "Strong":
            return "that efficiency tier would reflect a stronger conference tournament profile and may imply a more competitive overall postseason resume."
        elif tier == "Elite":
            return "that efficiency tier would reflect one of the strongest conference tournament profiles and may imply a very strong overall postseason resume."
        else:
            return "that efficiency tier would not suggest a meaningful conference tournament profile."

    # -----------------------
    # LOAD DATA
    # -----------------------
    data = pd.read_csv("MB_All_Conf.csv")

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    data = data.rename(columns={
        "conference_ranking": "conf_rank",
        "conf_win_percent": "conf_win_pct",
        "conf_win_percentage": "conf_win_pct"
    })

    data = data.loc[:, ~data.columns.duplicated()].copy()

    # ACC includes all teams in the conference tournament
    if "conf_tournament" not in data.columns:
        data["conf_tournament"] = 1

    required_cols = [
        "conference",
        "availability",
        "conf_win_pct",
        "conf_rank",
        "conf_tournament",
        "ncaa_tournament",
        "postseason_eff"
    ]

    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns:", data.columns.tolist())
        st.stop()

    # Clean numeric columns
    data["availability"] = clean_numeric(data["availability"], percent=True)
    data["conf_win_pct"] = clean_numeric(data["conf_win_pct"], percent=True)
    data["conf_rank"] = clean_numeric(data["conf_rank"])
    data["conf_tournament"] = clean_numeric(data["conf_tournament"])
    data["ncaa_tournament"] = clean_numeric(data["ncaa_tournament"])
    data["postseason_eff"] = clean_numeric(data["postseason_eff"])

    data = data.dropna(subset=required_cols).copy()

    data["conf_rank"] = data["conf_rank"].round().astype(int)
    data["conf_tournament"] = data["conf_tournament"].round().astype(int)
    data["ncaa_tournament"] = data["ncaa_tournament"].round().astype(int)

    # -----------------------
    # CONFERENCE FILTER
    # -----------------------
    conferences = sorted(data["conference"].dropna().unique())
    selected_conf = st.selectbox("Select Conference", conferences)

    filtered_data = data[data["conference"] == selected_conf].copy()

    if len(filtered_data) < 5:
        st.warning("Not enough data for this conference.")
        st.stop()

    # -----------------------
    # PREP TRAINING DATA
    # -----------------------
    X1 = filtered_data[["availability", "conf_win_pct"]].copy()
    y_conf_tourney = filtered_data["conf_tournament"].copy()
    y_ncaa = filtered_data["ncaa_tournament"].copy()
    y_rank = filtered_data["conf_rank"].copy()
    y_eff = filtered_data["postseason_eff"].copy()

    X1["availability"] = pd.to_numeric(X1["availability"], errors="coerce")
    X1["conf_win_pct"] = pd.to_numeric(X1["conf_win_pct"], errors="coerce")
    y_conf_tourney = pd.to_numeric(y_conf_tourney, errors="coerce")
    y_ncaa = pd.to_numeric(y_ncaa, errors="coerce")
    y_rank = pd.to_numeric(y_rank, errors="coerce")
    y_eff = pd.to_numeric(y_eff, errors="coerce")

    train_df = pd.concat(
        [
            X1,
            y_conf_tourney.rename("conf_tournament"),
            y_ncaa.rename("ncaa_tournament"),
            y_rank.rename("conf_rank"),
            y_eff.rename("postseason_eff")
        ],
        axis=1
    ).dropna().copy()

    if len(train_df) < 5:
        st.error("Not enough clean rows remain after data cleaning.")
        st.stop()

    X1 = train_df[["availability", "conf_win_pct"]].astype(float)
    y_conf_tourney = train_df["conf_tournament"].astype(int)
    y_ncaa = train_df["ncaa_tournament"].astype(int)
    y_rank = train_df["conf_rank"].astype(float)
    y_eff = train_df["postseason_eff"].astype(float)

    # -----------------------
    # TRAIN MODELS
    # -----------------------
    model_conf_tourney = RandomForestClassifier(random_state=42)
    model_ncaa = RandomForestClassifier(random_state=42)
    model_rank = RandomForestRegressor(random_state=42)
    model_eff = RandomForestRegressor(random_state=42)

    model_conf_tourney.fit(X1, y_conf_tourney)
    model_ncaa.fit(X1, y_ncaa)
    model_rank.fit(X1, y_rank)
    model_eff.fit(X1, y_eff)

    X2 = train_df[["ncaa_tournament", "conf_rank", "postseason_eff"]].astype(float)
    y_avail = train_df["availability"].astype(float)
    y_conf = train_df["conf_win_pct"].astype(float)

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

        availability_pct = st.number_input(
            "Availability (%)",
            min_value=0,
            max_value=100,
            value=90,
            step=1
        )

        conf_win_pct_input = st.number_input(
            "Conference Win % (%)",
            min_value=0,
            max_value=100,
            value=60,
            step=1
        )

        if st.button("Predict Forward"):
            availability = availability_pct / 100
            conf_win_pct = conf_win_pct_input / 100

            X_pred = pd.DataFrame([{
                "availability": availability,
                "conf_win_pct": conf_win_pct
            }])

            conf_tourney = int(model_conf_tourney.predict(X_pred)[0])
            ncaa = int(model_ncaa.predict(X_pred)[0])
            ncaa_prob = float(model_ncaa.predict_proba(X_pred)[0][1])
            rank = int(round(float(model_rank.predict(X_pred)[0])))
            eff = float(model_eff.predict(X_pred)[0])

            if eff < 0:
                eff = 0.0

            tier = get_tier(eff)
            tier_text = get_forward_tier_text(tier)

            st.write("### Results")
            st.write(f"**Conference:** {selected_conf}")
            st.write(f"**Conference Tournament Qualification:** {conf_tourney}")
            st.write(f"**NCAA Tournament Qualification:** {ncaa}")
            st.write(f"**NCAA Tournament Qualification Probability:** {round(ncaa_prob, 3)}")
            st.write(f"**Conference Rank:** {rank}")
            st.write(f"**Postseason Efficiency:** {round(eff, 3)}")
            st.write(f"**Efficiency Tier:** {tier}")

            st.markdown("### What This Means")
            st.info(
                f"This model predicts that a team with {availability_pct}% availability "
                f"and a {conf_win_pct_input}% conference win rate is likely to {'make' if conf_tourney == 1 else 'miss'} the conference tournament. "
                f"The team is projected to finish around {rank} in the conference. "
                f"The model also estimates a {round(ncaa_prob * 100, 1)}% probability of making the NCAA tournament. "
                f"With a postseason efficiency score of {round(eff, 3)}, this team is performing at a '{tier}' level. "
                f"{tier_text}"
            )

            st.markdown("### Metric Explanations")
            st.write(f"**Conference Tournament Qualification ({conf_tourney})**: Indicates whether the model predicts the team will make the conference tournament (1 = yes, 0 = no).")
            st.write(f"**NCAA Tournament Qualification ({ncaa})**: Indicates whether the model predicts the team will make the NCAA tournament (1 = yes, 0 = no).")
            st.write(f"**NCAA Tournament Qualification Probability ({round(ncaa_prob, 3)})**: The model's confidence that the team will make the NCAA tournament.")
            st.write(f"**Conference Rank ({rank})**: The expected final standing within the conference.")
            st.write(f"**Postseason Efficiency ({round(eff, 3)})**: A custom metric based on conference tournament postseason performance. Higher values suggest stronger performance in conference postseason play.")
            st.write(f"**Efficiency Tier ({tier})**: A category that makes the efficiency score easier to interpret.")

    # -----------------------
    # REVERSE MODE
    # -----------------------
    else:
        st.subheader(f"Reverse Prediction — {selected_conf}")

        ncaa_tournament = st.number_input(
            "NCAA Tournament (0 or 1)",
            min_value=0,
            max_value=1,
            value=1,
            step=1
        )

        conf_rank = st.number_input(
            "Conference Rank",
            min_value=1,
            max_value=25,
            value=5,
            step=1
        )

        postseason_eff = st.number_input(
            "Postseason Efficiency",
            min_value=0.0,
            max_value=5.0,
            value=1.5,
            step=0.01
        )

        if st.button("Predict Reverse"):
            X_pred = pd.DataFrame([{
                "ncaa_tournament": ncaa_tournament,
                "conf_rank": conf_rank,
                "postseason_eff": postseason_eff
            }])

            avail_pred = float(model_avail.predict(X_pred)[0])
            conf_pred = float(model_conf.predict(X_pred)[0])

            avail_pred = max(0.0, min(1.0, avail_pred))
            conf_pred = max(0.0, min(1.0, conf_pred))

            avail_pred_pct = int(round(avail_pred * 100))
            conf_pred_pct = int(round(conf_pred * 100))

            tier = get_tier(postseason_eff)
            tier_text = get_reverse_tier_text(tier)

            st.write("### Results")
            st.write(f"**Conference:** {selected_conf}")
            st.write(f"**Predicted Availability:** {avail_pred_pct}%")
            st.write(f"**Predicted Conference Win %:** {conf_pred_pct}%")
            st.write(f"**Efficiency Tier:** {tier}")

            st.markdown("### What This Means")
            st.info(
                f"To reach an NCAA tournament outcome of {ncaa_tournament} with a conference rank of {conf_rank}, "
                f"a team would likely need about {avail_pred_pct}% availability and about a "
                f"{conf_pred_pct}% conference win rate. "
                f"A postseason efficiency value in the '{tier}' range would indicate that {tier_text}"
            )

            st.markdown("### Metric Explanations")
            st.write(f"**Predicted Availability ({avail_pred_pct}%)**: The estimated player availability associated with this outcome profile.")
            st.write(f"**Predicted Conference Win % ({conf_pred_pct}%)**: The estimated in-conference win rate associated with this outcome profile.")
            st.write(f"**Efficiency Tier ({tier})**: The selected postseason efficiency translated into a performance category.")

    st.markdown("""
    ---
    ### Efficiency Interpretation (Basketball)

    - 0.00 → No Conference Tournament Appearance
    - 0.01–0.99 → Below Average
    - 1.00–1.49 → Average
    - 1.50–2.99 → Strong
    - 3.00+ → Elite

    These ranges are based on the observed postseason efficiency distribution in the dataset.
    """)


if __name__ == "__main__":
    main()
