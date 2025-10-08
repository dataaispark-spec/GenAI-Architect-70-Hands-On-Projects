# Course 2: Machine Learning Basics 📊

## Overview
This course covers the fundamentals of machine learning, essential for GenAI. We explore data handling, predictive algorithms, and model evaluation using Scikit-learn.

## Key Learning Objectives
- Understand ML types and applications
- Implement data preprocessing and splitting
- Build and evaluate linear regression models
- Apply classification techniques

## Core Topics

### 1. **Machine Learning Fundamentals**
- **What is ML?**: Pattern learning from data
- **Types**: Supervised (prediction), Unsupervised (clustering), Reinforcement
- **Key Process**: Data → Model → Prediction → Evaluation

### 2. **Data Handling & Preprocessing**
- **Data Splitting**: Train (70%), Validation (15%), Test (15%)
- **Features & Labels**: Structured datasets
- **Common Issues**: Overfitting, data bias, missing values

### 3. **Linear Regression**
- **Purpose**: Predict continuous values
- **Math**:  y = w·x + b
- **Training**: Minimi ze Mean Squared Error
- **Optimization**: Gradient descent (w := w - α·∂J/∂w)

**Implementation Example (Scikit-learn):**
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression().fit(X, y)
print(f"Slope: {model.coef_[0]:.2f}")  # Approx 2.0
print(f"Intercept: {model.intercept_:.2f}")  # Approx 0.0
prediction = model.predict([[6]])[0]
```

### 4. **Classification Basics**
- **Binary Classification**: Decide categories (e.g., spam/not spam)
- **Logistic Regression**: Probability-based classifier
- **Decision Boundaries**: Linear separation in feature space

### 5. **Model Evaluation**
- **Regression Metrics**: MAE, RMSE, R² score
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Cross-Validation**: Prevent overfitting

## Practical Applications
- **Finance**: Stock price prediction, credit risk assessment
- **Healthcare**: Disease diagnosis, treatment outcomes
- **Retail**: Sales forecasting, customer segmentation

## Hands-on Experience
- **Labs**: Build models with real datasets in Jupyter notebooks
- **Tools**: Scikit-learn, pandas, matplotlib for data analysis
- **Projects**: End-to-end ML pipeline from data to deployment

## Architect Perspective
- **Data-first Approach**: Clean, quality data drives ML success
- **Model Lifecycle**: Train → Validate → Deploy → Monitor
- **Scalability**: Handle large datasets efficiently
- **Ethics**: Address bias and fairness in ML systems

## Learning Outcomes
- ✅ Grasp supervised and unsupervised ML
- ✅ Preprocess and split datasets properly
- ✅ Train linear regression models
- ✅ Evaluate ML model performance
- ✅ Build basic ML prototypes

## Resources
- **Scikit-learn Docs**: https://scikit-learn.org
- **Hands-on ML Book**: https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

---

*Estimated Study Time: 8-10 hours* | *Prerequisites: Python Basics*

## Next Up
**Course 3: Neural Networks Fundamentals** - Understanding artificial neurons and deep learning basics.

*Ready to transform raw data into intelligent predictions! 🚀*
