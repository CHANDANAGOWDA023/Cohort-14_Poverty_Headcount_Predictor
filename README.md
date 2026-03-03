# Cohort-14_Poverty_Headcount_Predictor

# 📊 Poverty Dataset Integration

## Overview

This dataset was created by integrating district-level poverty and socio-economic data from the following sources:

1. **India District MPI 2018 Table 5a.1** –

   * Multidimensional Poverty Index (MPI)
   * Headcount Ratio (H)

2. **India District MPI 2018 Table 5a.4** – 

   * Electricity Deprivation (%)

3. `district_wise_mpi.csv` –

   * Total Population
   * Households
   * Total Working Population

All datasets were merged using **State** and **District** as common identifiers.
---
## Feature Engineering

The following features were derived:

* **Average Household Size** = Total Population / Households
* **Electricity Access (%)** = 100 − Electricity_Deprivation
* **Average Annual Income (Proxy)**
  = (Total Working Population / Total Population) × ₹1,50,000

---

## Outlier Handling

Income outliers were treated using **IQR-based Winsorization**:

* Values outside *(Q1 − 1.5×IQR, Q3 + 1.5×IQR)* were clipped.

---

## Final Dataset

**File Name:** `integrated_poverty_dataset.csv`
**Total Records:** 576 districts

### Columns Included:

* State, District
* Total Population
* Households
* Total Working Population
* Electricity_Deprivation
* Average Household Size
* Electricity Access
* Average Annual Income
* MPI (Target 1)
* Headcount_Ratio (H) (Target 2)

The dataset is now cleaned, integrated, feature-engineered, and ready for regression modeling.

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
