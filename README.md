# College Basketball Team Prediction App

This project is a machine learning web application that models relationships between team performance metrics and postseason outcomes in men's college basketball. The app allows users to generate forward and reverse predictions within specific conferences or across all conferences using real data from the 2025–2026 season.

---

## Overview

The application predicts:

- Conference tournament qualification (0 = No, 1 = Yes)
- NCAA tournament qualification (0 = No, 1 = Yes)
- Probability of making the NCAA tournament
- Conference rank
- Postseason efficiency
- Efficiency tier classification

In this model:
- **Conference tournament qualification** represents whether a team makes the conference tournament
- **NCAA tournament qualification** is modeled separately
- **Postseason efficiency is based on conference tournament performance**

The model does not define NCAA tournament selection through efficiency alone, though stronger efficiency may imply a stronger overall postseason resume.

The app also includes a comparison feature that identifies real teams with similar projected profiles.

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
- Two similar real teams based on predicted profile

### Reverse Prediction
Inputs:
- NCAA Tournament Qualification
- Conference Rank
- Postseason Efficiency

Outputs:
- Predicted Availability
- Predicted Conference Win Percentage
- Efficiency Tier

### Conference Scope
Users can run predictions within:
- A single conference
- **All conferences combined**

---

## Data Sources

- Data was compiled from official athletics websites for the Power 4 conferences (ACC, Big Ten, Big 12, SEC) during the 2025–2026 men's college basketball season
- Conference tournament and NCAA tournament indicators were included as separate variables
- Metrics were aligned across sources to ensure consistency

---

## Postseason Efficiency Metric

Postseason efficiency is a custom metric based on conference tournament performance.

Formula:

Efficiency = Weighted Win % × log(Total Postseason Weight)

Where:
- Weighted Win % = conference tournament success adjusted for game importance
- Total Postseason Weight = cumulative importance of conference tournament games

This metric rewards teams that:
- Perform well in conference postseason play
- Sustain performance across higher-impact conference tournament games

Higher efficiency values indicate stronger conference postseason performance and may imply a more competitive overall postseason resume.

---

## Efficiency Tier Classification

| Range | Tier |
|------|------|
| 0.00 | No Conference Tournament Appearance |
| 0.01 – 0.99 | Below Average |
| 1.00 – 1.74 | Average |
| 1.75 – 2.74 | Strong |
| 2.75+ | Elite |

These ranges are based on the observed postseason efficiency distribution in the dataset.

---

## Model Approach

The app uses Random Forest models:

### Classification
- Conference tournament qualification
- NCAA tournament qualification

### Regression
- Conference rank
- Postseason efficiency
- Reverse predictions for availability and conference win percentage

---

## Limitations

- Predictions are based on historical patterns and are not guarantees
- NCAA tournament selection is influenced by factors beyond this model
- Reverse predictions are approximate
- Efficiency is a custom metric and may not capture all performance factors
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
- Machine learning classification and regression
- Model interpretation
- Applied sports analytics
- Similarity-based team comparison
- End-to-end deployment

---

## Live App

(https://collegembbteam-prediction-app-hngu834eila4zrxx93bont.streamlit.app/)
