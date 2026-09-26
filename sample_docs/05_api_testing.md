# Prediction APIs and tests

A prediction API should validate input types, required fields and acceptable ranges. Malformed requests should return a clear client error instead of an internal stack trace. Keep training separate from inference so that the same fitted preprocessing and model are reused for predictions.

Tests should cover valid input, empty input, missing fields and unsupported formats. Integration tests should check the full path from a request to its output. A Flask test client can exercise routes without starting a public server. A local development server is not a production deployment.
