# College Basketball Team Prediction App

This project is a machine learning web application that models relationships between team performance metrics and postseason outcomes in men's college basketball. The app allows users to simulate forward and reverse predictions across conferences using real data from the 2025–2026 season.

---

## Overview

The application predicts:

- Postseason qualification (0 = No, 1 = Yes)
- Probability of making the postseason
- Conference rank
- Postseason efficiency
- Efficiency tier classification

It also supports reverse predictions, estimating performance inputs based on desired outcomes.

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

- Data was compiled from official athletics websites for the **Power 4 conferences (ACC, Big Ten, Big 12, SEC)** during the 2025–2026 men's college basketball season
- Postseason performance data was aggregated and standardized across conferences
- Postseason results were weighted based on game importance (e.g., conference tournaments vs NCAA tournament)
- Metrics were aligned across sources to ensure consistency

---

## Postseason Efficiency Metric

Postseason efficiency is a custom metric designed to capture both performance and level of competition.

It is defined as: Efficiency = Weighted Win % × log(Total Postseason Weight)


Where:
- **Weighted Win %** = postseason success adjusted for game importance  
- **Total Weight** = cumulative importance of postseason games  

This metric rewards teams that:
- Perform well
- Sustain performance across higher-stakes games

---

## Efficiency Tier Classification

Efficiency values are categorized into tiers based on observed data distribution:

| Range | Tier |
|------|------|
| 0.00 | No Postseason Appearance |
| 0.01 – 0.99 | Below Average |
| 1.00 – 1.74 | Average |
| 1.75 – 2.74 | Strong |
| 2.75+ | Elite |

These ranges are specific to the basketball dataset.

---

## Model Approach

The app uses Random Forest models for both classification and regression:

- Classification:
  - Predicts postseason qualification

- Regression:
  - Predicts conference rank
  - Predicts postseason efficiency
  - Reverse models estimate availability and conference win %

---

## Limitations

- Predictions are based on historical patterns and should be interpreted as estimates
- Reverse models are less reliable for certain variables
- Efficiency is a custom metric and may not capture all aspects of team performance
- Differences in conference reporting may affect comparability

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Google Sheets (data preparation)

---

## Purpose

This project demonstrates:

- Feature engineering using a custom efficiency metric
- Supervised machine learning (classification and regression)
- Model interpretation and applied analytics
- End-to-end workflow from raw data to deployed application

---

## Live App

(https://collegembbteam-prediction-app-hngu834eila4zrxx93bont.streamlit.app/)

---
