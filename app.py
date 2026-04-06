import pandas as pd
import numpy as np

# -----------------------------------
# CONFIG
# -----------------------------------
STANDINGS_FILE = "acc_standings_summary.csv"
SCRAPED_AVAILABILITY_FILE = "acc_team_availability.csv"
POSTSEASON_FILE = "acc_postseason_scraped.csv"
NCAA_TOURNAMENT_FILE = "acc_ncaa_tournament_flags.csv"
OUTPUT_FILE = "acc_final_automated_summary.csv"
ROUND_DECIMALS = 4

ROUND_WEIGHTS = {
    "Pre 1st Round": 2,
    "1st Round": 2,
    "2nd Round": 6,
    "Quarterfinals": 14,
    "Semifinals": 30,
    "Conference Championship": 62,
}


# -----------------------------------
# LOAD DATA
# -----------------------------------
def load_data(standings_file, postseason_file, scraped_availability_file, ncaa_tournament_file):
    standings = pd.read_csv(standings_file)
    scraped_availability = pd.read_csv(scraped_availability_file)
    postseason_games = pd.read_csv(postseason_file)
    ncaa_flags = pd.read_csv(ncaa_tournament_file)
    return standings, postseason_games, scraped_availability, ncaa_flags


# -----------------------------------
# CLEAN TEAM NAMES
# -----------------------------------
def normalize_team_names(df, team_col):
    df = df.copy()

    team_name_map = {
        "Florida St.": "Florida State",
        "Miami": "Miami (FL)",
        "Pitt": "Pittsburgh",
    }

    df[team_col] = df[team_col].replace(team_name_map)
    return df


# -----------------------------------
# BUILD TEAM BASE FROM SCRAPED SOURCES
# -----------------------------------
def build_team_base(standings, scraped_availability, ncaa_flags):
    standings = normalize_team_names(standings, "team")
    scraped_availability = normalize_team_names(scraped_availability, "Team")
    ncaa_flags = normalize_team_names(ncaa_flags, "team")

    scraped_availability = scraped_availability.rename(
        columns={"Team": "team", "avg_availability": "availability"}
    )

    team_base = standings.merge(
        scraped_availability[["team", "availability"]],
        on="team",
        how="left"
    )

    team_base = team_base.merge(
        ncaa_flags[["team", "ncaa_tournament"]],
        on="team",
        how="left"
    )

    return team_base


# -----------------------------------
# STANDARDIZE POSTSEASON INPUT
# expects Team, Round, Result
# -----------------------------------
def standardize_postseason_input(postseason_games):
    post = postseason_games.copy()

    required_cols = ["Team", "Round", "Result"]
    for col in required_cols:
        if col not in post.columns:
            raise ValueError(f"Missing required column: {col}")

    post = post[["Team", "Round", "Result"]].copy()
    post = normalize_team_names(post, "Team")

    post["Round"] = post["Round"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    post["Result"] = post["Result"].astype(str).str.strip().str.upper()

    return post


# -----------------------------------
# APPLY ROUND WEIGHTS
# -----------------------------------
def apply_round_weights(post):
    post = post.copy()
    post["Weight"] = post["Round"].map(ROUND_WEIGHTS)

    if post["Weight"].isna().any():
        missing_rounds = sorted(post.loc[post["Weight"].isna(), "Round"].dropna().unique())
        raise ValueError(f"These rounds are missing from ROUND_WEIGHTS: {missing_rounds}")

    return post


# -----------------------------------
# CALCULATE POSTSEASON METRICS
# win = full weight
# loss = half weight
# -----------------------------------
def calculate_postseason_metrics(post):
    post = post.copy()

    post["weighted_postseason_points"] = np.where(
        post["Result"] == "W",
        post["Weight"],
        post["Weight"] / 2
    )

    post_team = (
        post.groupby("Team", as_index=False)
        .agg(
            total_weight=("Weight", "sum"),
            weighted_postseason_points=("weighted_postseason_points", "sum"),
        )
    )

    post_team["weighted_postseason_win_pct"] = (
        post_team["weighted_postseason_points"] / post_team["total_weight"]
    ).fillna(0)

    post_team["postseason_eff"] = (
        post_team["weighted_postseason_win_pct"] * np.log(post_team["total_weight"])
    ).fillna(0)

    post_team = post_team.rename(columns={"Team": "team"})
    return post_team


# -----------------------------------
# ASSIGN EFFICIENCY TIER
# -----------------------------------
def assign_eff_tier(post_team):
    post_team = post_team.copy()

    def get_eff_tier(x):
        if x == 0:
            return "No Postseason Appearance"
        elif x < 1.0:
            return "Below Average"
        elif x < 1.5:
            return "Average"
        elif x < 3.0:
            return "Strong"
        else:
            return "Elite"

    post_team["eff_tier"] = post_team["postseason_eff"].apply(get_eff_tier)
    return post_team


# -----------------------------------
# BUILD FINAL SUMMARY
# -----------------------------------
def build_final_summary(team_base, post_team):
    generated = team_base.merge(
        post_team[
            [
                "team",
                "weighted_postseason_points",
                "weighted_postseason_win_pct",
                "postseason_eff",
                "eff_tier",
            ]
        ],
        on="team",
        how="left"
    )

    generated["availability"] = generated["availability"].fillna(0)
    generated["ncaa_tournament"] = generated["ncaa_tournament"].fillna(0).astype(int)
    generated["weighted_postseason_points"] = generated["weighted_postseason_points"].fillna(0)
    generated["weighted_postseason_win_pct"] = generated["weighted_postseason_win_pct"].fillna(0)
    generated["postseason_eff"] = generated["postseason_eff"].fillna(0)
    generated["eff_tier"] = generated["eff_tier"].fillna("No Postseason Appearance")

    return generated.sort_values(["conference_ranking", "team"]).reset_index(drop=True)


# -----------------------------------
# ROUND OUTPUT COLUMNS
# -----------------------------------
def round_output_columns(generated, decimals=4):
    generated = generated.copy()

    columns_to_round = [
        "win_percent",
        "conf_win_percent",
        "availability",
        "weighted_postseason_points",
        "weighted_postseason_win_pct",
        "postseason_eff",
    ]

    for col in columns_to_round:
        generated[col] = generated[col].round(decimals)

    return generated


# -----------------------------------
# SAVE OUTPUT
# -----------------------------------
def save_output(generated, output_file):
    generated.to_csv(output_file, index=False)


# -----------------------------------
# MAIN
# -----------------------------------
def main():
    standings, postseason_games, scraped_availability, ncaa_flags = load_data(
        STANDINGS_FILE,
        POSTSEASON_FILE,
        SCRAPED_AVAILABILITY_FILE,
        NCAA_TOURNAMENT_FILE
    )

    team_base = build_team_base(standings, scraped_availability, ncaa_flags)
    post = standardize_postseason_input(postseason_games)
    post = apply_round_weights(post)
    post_team = calculate_postseason_metrics(post)
    post_team = assign_eff_tier(post_team)

    generated = build_final_summary(team_base, post_team)
    generated = round_output_columns(generated, decimals=ROUND_DECIMALS)
    save_output(generated, OUTPUT_FILE)

    print("\nFINAL AUTOMATED SUMMARY (first 10 rows):")
    print(generated.head(10))
    print(f"\nDone. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
