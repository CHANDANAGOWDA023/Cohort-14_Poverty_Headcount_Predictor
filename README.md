# Cohort-14_Poverty_Headcount_Predictor
---

## Individual Contribution – Srushti A T

### Outlier Handling (IQR + Winsorization)

As part of the preprocessing stage, extreme values in the **Average Annual Income** feature were handled using the IQR method and Winsorization (clipping).

#### Why this method was chosen:
- Income data is typically skewed and does not follow normal distribution.
- IQR is robust for skewed data and does not assume normality.
- It identifies extreme values based on the middle 50% of the dataset.

#### Why Winsorization was applied:
- Instead of removing districts, extreme values were capped at IQR bounds.
- Preserved dataset size and prevented loss of important information.
- Reduced distortion in regression model training.

#### Impact on Model:
- Stabilized income distribution.
- Reduced sensitivity to extreme cases.
- Improved overall model reliability and prediction performance.