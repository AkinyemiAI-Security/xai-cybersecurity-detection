

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Dataset
data = {
    'url_length': [10, 50, 70, 15, 80, 25, 90, 45],
    'has_https': [1, 0, 0, 1, 0, 1, 0, 1],
    'num_dots': [1, 3, 4, 1, 5, 2, 6, 2],
    'has_at_symbol': [0, 1, 1, 0, 1, 0, 1, 0],
    'label': [0, 1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['url_length', 'has_https', 'num_dots', 'has_at_symbol']]
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Feature importance (XAI)
importance = model.feature_importances_

features = X.columns

print("Feature Importance (Explainability):")
for i, v in enumerate(importance):
    print(f"{features[i]}: {v:.3f}")

# Test prediction
test_url = [[65, 0, 4, 1]]
prediction = model.predict(test_url)

print("\nPrediction:")
if prediction[0] == 1:
    print("⚠️ Phishing detected")
else:
    print("✅ Safe")

print("\nExplanation:")
for i, v in enumerate(importance):
    print(f"{features[i]} contributed {v:.3f} to the decision")
