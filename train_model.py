import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv('data/vgsales.csv')

# Preprocessing
data = data.dropna(subset=['Global_Sales'])
X = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
y = data['Global_Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/model.pkl')
print("Model saved successfully!")
