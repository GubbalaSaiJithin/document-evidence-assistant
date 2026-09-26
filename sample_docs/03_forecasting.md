# Demand forecasting

Mean absolute error, or MAE, measures the average absolute forecast error in the target's units. Root mean squared error, or RMSE, penalizes large errors more heavily. Compare a forecast model with a seasonal baseline such as the previous day's or previous week's value for the same hour.

A rolling one-hour-ahead evaluation can use the latest observed demand before each prediction. A day-ahead forecast cannot use actual demand observed later in that day. Use chronological validation or expanding time windows. Missing hours should be represented explicitly before applying lag operations; do not assume every row is exactly one hour apart.
