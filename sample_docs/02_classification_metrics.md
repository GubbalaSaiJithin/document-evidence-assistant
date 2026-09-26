# Evaluating classification models

Accuracy is the fraction of correct predictions. It can hide poor minority-class performance. Precision measures how many predicted positives are correct. Recall measures how many actual positives are found. F1 is the harmonic mean of precision and recall. Macro F1 gives each class equal weight.

Use a confusion matrix to inspect the errors between classes. Compare with a simple baseline. Select models and thresholds using validation data, then report results on a held-out test set. A probability output is not necessarily calibrated. Reliability diagrams and calibration methods help assess the relationship between predicted probabilities and observed outcomes.
