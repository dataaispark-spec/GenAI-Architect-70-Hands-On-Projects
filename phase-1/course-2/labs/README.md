# Course 2: Machine Learning Basics - Hands-on Labs 🛠️

## Lab Overview: Linear Regression Model Building 🧪

This lab is your first hands-on dive into machine learning! You'll build a linear regression model from scratch using Scikit-learn, split datasets properly, and evaluate your model's performance. Perfect for beginners transitioning from theory to practice. ⬇️

---

## Prerequisites
- Python installed and basic knowledge (from Course 1)
- Scikit-learn library: Install with `pip install scikit-learn`
- Matplotlib for visualization: `pip install matplotlib`
- numpy: `pip install numpy`
- Jupyter Notebook (for interactive coding)

## 🏗️ **Lab Objectives**
- Split data into train/validation/test sets
- Build and train a linear regression model
- Make predictions and visualize results
- Evaluate model performance with metrics

## 📋 **Step-by-Step Instructions**

### Step 1: Setup Environment 🌟
Create a new Jupyter notebook called `course2_linear_regression.ipynb`

Import required libraries:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Enable inline plotting for Jupyter
%matplotlib inline
```

### Step 2: Generate Sample Dataset 📊
Create a realistic dataset to work with:
```python
# Generate synthetic house price data
np.random.seed(42)  # For reproducibility

# Features: house size (sq ft), number of bedrooms
n_samples = 1000
sizes = np.random.uniform(800, 3500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)

# True relationship: price = 50 * size + 30 * bedrooms * size/1000 + noise
prices = 50 * sizes + 30 * bedrooms * (sizes/1000) + np.random.normal(0, 5000, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'size': sizes,
    'bedrooms': bedrooms,
    'price': prices
})

print(data.head())
print(f"Dataset shape: {data.shape}")
```

### Step 3: Explore Data 🔍
Understand your data before modeling:
```python
# Basic statistics
print(data.describe())

# Visualize relationships
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(data['size'], data['price'], alpha=0.5)
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Price vs Size')

plt.subplot(1, 2, 2)
plt.scatter(data['bedrooms'], data['price'], alpha=0.5)
plt.xlabel('Bedrooms')
plt.ylabel('Price ($)')
plt.title('Price vs Bedrooms')

plt.tight_layout()
plt.show()
```

### Step 4: Data Splitting and Preprocessing 🎯
Proper data splitting is crucial for ML:
```python
# Features and target
X = data[['size', 'bedrooms']]
y = data['price']

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 0.176*0.85 ≈ 0.15

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
```

### Step 5: Build and Train Model 🚀
Train your linear regression model:
```python
# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Print coefficients
print(f"Intercept: ${model.intercept_:,.0f}")
print(f"Coefficient for size: ${model.coef_[0]:,.2f} per sq ft")
print(f"Coefficient for bedrooms: ${model.coef_[1]:,.2f} per bedroom")
```

### Step 6: Make Predictions and Visualize 📈
See how well your model predicts:
```python
# Predict on validation set
y_val_pred = model.predict(X_val)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.7, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices (Validation Set)')
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 7: Evaluate Model Performance 📊
Quantify your model's accuracy:
```python
def evaluate_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\\n{dataset_name} Set Evaluation:")
    print(f"MAE: ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R² Score: {r2:.3f} ({r2*100:.1f}% variance explained)")
    
    return mae, rmse, r2

# Evaluate on different sets
train_val_pred = model.predict(X_train)
val_mae, val_rmse, val_r2 = evaluate_model(y_val, y_val_pred, "Validation")

test_pred = model.predict(X_test)
test_mae, test_rmse, test_r2 = evaluate_model(y_test, test_pred, "Test")
```

### Step 8: Experiment and Improve 🔬
Try improving your model:
```python
# Option 1: Feature engineering - add interaction term
X_train_engineered = X_train.copy()
X_val_engineered = X_val.copy()
X_test_engineered = X_test.copy()

# Interaction: size * bedrooms (luxury factor)
X_train_engineered['size_bedroom'] = X_train['size'] * X_train['bedrooms']
X_val_engineered['size_bedroom'] = X_val['size'] * X_val['bedrooms']
X_test_engineered['size_bedroom'] = X_test['size'] * X_test['bedrooms']

# Train new model
model_engineered = LinearRegression()
model_engineered.fit(X_train_engineered, y_train)

# Compare performance
y_val_engineered_pred = model_engineered.predict(X_val_engineered)
_, engineered_rmse, engineered_r2 = evaluate_model(y_val, y_val_engineered_pred, "Validation (Engineered)")

print(f"\\nImprovement with feature engineering:")
print(f"R² improved by {engineered_r2 - val_r2:.3f}")
```

---

## 🤔 **Lab Questions & Reflections**

### Conceptual Questions
1. **Data Splitting**: Why do we need separate train, validation, and test sets? What happens if we don't?
2. **Overfitting**: If your model performs much better on train than test, what might be happening?
3. **R² Score**: What does an R² of 0.85 mean? What about 0.2?

### Technical Questions
1. **Coefficient Interpretation**: How do you interpret the coefficient for house size?
2. **Outliers**: What effect might outliers have on linear regression?
3. **Feature Selection**: Why did adding the interaction term improve performance?

### Practical Questions
1. **Model Evaluation**: Based on your metrics, is your model ready for production?
2. **Data Quality**: What data quality issues might be present?
3. **Next Steps**: What could you do to further improve performance?

---

## 📚 **Additional Resources**

- **Scikit-learn Documentation**: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **Linear Regression Deep Dive**: [https://towardsdatascience.com/simple-linear-regression-guide-86d9b9aecd2f](https://towardsdatascience.com/simple-linear-regression-guide-86d9b9aecd2f)
- **Evaluating Regression Models**: [https://www.statisticssolutions.com/evaluating-regression-models/](https://www.statisticssolutions.com/evaluating-regression-models/)

---

## ✅ **Lab Completion Checklist**

- [ ] Created new Jupyter notebook
- [ ] Imported all required libraries
- [ ] Generated synthetic dataset
- [ ] Exploratory data analysis and visualization
- [ ] Properly split data into train/val/test
- [ ] Trained linear regression model
- [ ] Made predictions and evaluated performance
- [ ] Experimented with feature engineering
- [ ] Answered lab questions
- [ ] Saved notebook with outputs

**Congratulations! 🎉** You've built your first ML model. This foundation will carry you through all future GenAI work.

*Estimated completion time: 2-3 hours*
