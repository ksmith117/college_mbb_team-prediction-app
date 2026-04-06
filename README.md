# College Basketball Team Prediction App

This project is a machine learning web application that models relationships between team performance metrics and postseason outcomes in men’s college basketball. The app allows users to generate forward and reverse predictions across multiple conferences using real data from the 2025–2026 season.

---

## Overview

The application predicts:

- Conference Tournament Qualification (0 = No, 1 = Yes)
- NCAA Tournament Qualification (0 = No, 1 = Yes)
- NCAA Tournament Qualification Probability
- Conference Rank
- Postseason Efficiency
- Efficiency Tier classification

In this model, **postseason efficiency reflects conference tournament performance**, while NCAA tournament qualification is modeled separately.

---

## Features

### Forward Prediction
Inputs:
- Availability
- Conference Win Percentage

Outputs:
- Conference Tournament Qualification
- NCAA Tournament Qualification
- NCAA Tournament Qualification Probability
- Conference Rank
- Postseason Efficiency
- Efficiency Tier

---

### Reverse Prediction
Inputs:
- NCAA Tournament Qualification
- Conference Rank
- Postseason Efficiency

Outputs:
- Predicted Availability
- Predicted Conference Win Percentage
- Efficiency Tier

---

## Data Sources

- Data was compiled from official athletics websites for Power 4 conferences (ACC, Big Ten, Big 12, SEC) during the 2025–2026 men’s college basketball season
- Availability data was derived from injury and player status reports
- Conference standings and records were collected from official conference sources
- Conference tournament results were manually aggregated and standardized
- NCAA tournament participation was recorded as a binary indicator (1 = made tournament, 0 = did not)

---

## Postseason Efficiency Metric

Postseason efficiency is a custom metric designed to capture both performance and level of competition.

Formula:

Efficiency = Weighted Postseason Win % × log(Total Postseason Weight)

Where:
- Weighted Postseason Win % = postseason success adjusted by game importance  
- Total Postseason Weight = cumulative importance of tournament rounds  

Weights are assigned based on round importance (e.g., Championship > Semifinals > Quarterfinals).

This metric rewards teams that:
- Advance deeper in tournaments  
- Perform well in high-impact games  

Higher efficiency values are consistent with stronger postseason performance.

---

## Efficiency Tier Classification

| Range | Tier |
|------|------|
| 0.00 | No Conference Tournament Appearance |
| 0.01 – 0.99 | Below Average |
| 1.00 – 1.49 | Average |
| 1.50 – 2.99 | Strong |
| 3.00+ | Elite |

---

## Model Performance

### NCAA Tournament Classifier
- Accuracy: ~0.89  
- ROC AUC: ~0.94  

### Conference Rank Regressor
- Strong predictive performance based on availability and win %

### Postseason Efficiency Model
- Captures general trends in tournament success  
- Less stable due to variability in postseason results  

### Reverse Models
- Provide approximate estimates of required performance levels  
- Best used for interpretation, not exact prediction  

---

## Limitations

- Predictions are based on historical patterns and are not guarantees  
- Reverse predictions are approximate  
- Efficiency is a custom metric and may not capture all factors  
- Manual data aggregation may introduce minor inconsistencies  
- Differences across conferences may affect comparability  

---

## Tech Stack

- Python  
- Streamlit  
- Pandas  
- Scikit-learn  

---

## Purpose

This project demonstrates:

- Feature engineering using a custom metric  
- Classification and regression modeling  
- Model evaluation and interpretation  
- Applied sports analytics  
- End-to-end deployment with Streamlit  

---

## Live App

https://collegembbteam-prediction-app-hngu834eila4zrxx93bont.streamlit.app/
