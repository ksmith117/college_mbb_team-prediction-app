# College Football Team Prediction App

This project is a machine learning web application that models relationships between team performance metrics and postseason outcomes in college football. The app allows users to generate forward and reverse predictions within specific conferences using real data from the 2025–2026 season.

---

## Overview

The application predicts:

- Postseason qualification (0 = No, 1 = Yes)
- Probability of making the postseason
- Conference rank
- Postseason efficiency
- Efficiency tier classification

In this model, **postseason qualification represents making a bowl game**.

---

## Features

### Forward Prediction
Inputs:
- Availability
- Conference Win Percentage

Outputs:
- Postseason Qualification
- Postseason Qualification Probability
- Conference Rank
- Postseason Efficiency
- Efficiency Tier

---

### Reverse Prediction
Inputs:
- Postseason Qualification
- Conference Rank
- Postseason Efficiency

Outputs:
- Predicted Availability
- Predicted Conference Win Percentage
- Efficiency Tier

---

## Data Sources

- Data was compiled from official athletics websites for the Power 4 conferences (ACC, Big Ten, Big 12, SEC) during the 2025–2026 college football season
- Postseason performance data was aggregated and standardized across conferences
- Metrics were aligned across sources to ensure consistency

---

## Postseason Efficiency Metric

Postseason efficiency is a custom metric designed to capture both performance and level of competition.

Formula:

Efficiency = Weighted Win % × log(Total Postseason Weight)

Where:
- Weighted Win % = postseason success adjusted for game importance
- Total Postseason Weight = cumulative importance of postseason opportunities

This metric rewards teams that:
- Perform well in postseason settings
- Sustain performance across higher-impact games

Higher efficiency values are more consistent with teams that compete for conference championships and the College Football Playoff.

---

## Efficiency Tier Classification

| Range | Tier |
|------|------|
| 0.00 | No Postseason Appearance |
| 0.01 – 0.74 | Below Average |
| 0.75 – 1.24 | Average |
| 1.25 – 1.99 | Strong |
| 2.00+ | Elite |

---

## Model Performance

### Postseason Classifier
- Accuracy: 0.857
- Precision: 0.889
- Recall: 0.889
- F1 Score: 0.889
- ROC AUC: 0.933

### Conference Rank Regressor
- MAE: 1.246
- RMSE: 1.447
- R²: 0.799

### Postseason Efficiency Regressor
- MAE: 0.126
- RMSE: 0.223
- R²: -0.111

### Reverse Models

Conference Win %:
- MAE: 0.057
- RMSE: 0.065
- R²: 0.901

Availability:
- MAE: 0.036
- RMSE: 0.050
- R²: -0.797

---

## Limitations

- Predictions are based on historical patterns and are not guarantees
- Reverse predictions are approximate
- Efficiency is a custom metric and may not capture all factors
- Differences in conference reporting may affect consistency

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Google Sheets

---

## Purpose

This project demonstrates:
- Feature engineering using a custom metric
- Classification and regression modeling
- Model evaluation and interpretation
- Applied sports analytics
- End-to-end deployment workflow

---

## Live App

(https://collegembbteam-prediction-app-hngu834eila4zrxx93bont.streamlit.app/)
