# Development Principles

## Single Source of Truth for Feature Engineering

Feature engineering must have one authoritative implementation.

The production feature engineering function is:

`engineer_features()`

located in:

`src/prediction_pipeline.py`

Other components of the system, including explainability, must reuse this function rather than duplicate the feature engineering logic.

### Why?

Duplicating feature engineering creates a risk of inconsistency.

If the feature engineering logic is changed in `prediction_pipeline.py` but not in another module, the model prediction and explanation could be based on different features.

### Project Rule

> Feature engineering should be implemented once and reused throughout the production system.

Use:

```python
from src.prediction_pipeline import engineer_features