# Outlier Handling – IQR + Winsorization

## Objective
To reduce the impact of extreme income values on model training.

## Method Used
- IQR method to detect outliers.
- Winsorization (clipping) to cap extreme values.

## Reason for Selection
- Suitable for skewed economic data.
- Does not assume normal distribution.
- Preserves dataset size.

## Impact on Model
- Improved data stability.
- Reduced prediction error.
- Enhanced model reliability.