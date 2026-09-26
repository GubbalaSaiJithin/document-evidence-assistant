# Preventing data leakage

Data leakage occurs when training uses information unavailable at prediction time. Split the dataset before fitting preprocessing. Fit imputers, scalers and text vectorizers on training data only, then transform validation and test data using the fitted objects. A scikit-learn Pipeline helps keep preprocessing and model fitting together during cross-validation.

For time series, reserve later timestamps for evaluation and build lag features from strictly earlier observations. In bike demand data, casual and registered counts add up to the target total; they are unsuitable predictors when forecasting that same hour. Deduplicate examples before splitting or group related examples so near-duplicates do not cross partitions.
