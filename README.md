# Task 7: Support Vector Machines (SVM)

## Objective
Implement Support Vector Machines (SVM) for both linear and non-linear classification using Scikit-learn. Visualize decision boundaries, tune hyperparameters, and evaluate performance using cross-validation.

---

## Project Overview

This project demonstrates:

- Binary classification using SVM
- Linear Kernel SVM
- RBF (Radial Basis Function) Kernel SVM
- Decision boundary visualization (2D dataset)
- Hyperparameter tuning (C and gamma)
- Cross-validation for robust evaluation

---

## Tools & Libraries

- Python
- NumPy
- Matplotlib
- Scikit-learn

---

## Dataset

We use a synthetic binary classification dataset generated using:

`sklearn.datasets.make_classification()`

Dataset characteristics:
- 300 samples
- 2 informative features (for 2D visualization)
- Binary target classes (0 and 1)

---

## Workflow

### 1. Data Preparation
- Generate dataset
- Train-test split (80-20)
- Feature scaling using `StandardScaler`

### 2. Model Training

Linear SVM:
`SVC(kernel='linear', C=1)`

RBF SVM:
`SVC(kernel='rbf', C=1, gamma=0.1)`

### 3. Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report
- 5-Fold Cross Validation

### 4. Hyperparameter Tuning

GridSearchCV is used to optimize:
- C values: [0.1, 1, 10, 100]
- gamma values: [0.01, 0.1, 1]
- kernel: rbf

---

## Results

| Model        | Performance Insight |
|-------------|--------------------|
| Linear SVM  | Effective when data is linearly separable |
| RBF SVM     | Handles non-linear patterns better |

Cross-validation ensures the model generalizes well and avoids overfitting.

---

## Key Concepts

### Support Vector Machine (SVM)
SVM finds the optimal hyperplane that maximizes the margin between classes.

### Important Hyperparameters

C (Regularization Parameter)
- Large C → Low bias, high variance
- Small C → High bias, smoother margin

Gamma (RBF Kernel only)
- High gamma → Complex boundary, risk of overfitting
- Low gamma → Smoother decision boundary

---

## How to Run

1. Clone the repository:
```
git clone <your-repo-link>
cd <repo-folder>
```

2. Install dependencies:
```
pip install numpy matplotlib scikit-learn
```

3. Run the script:
```
python svm_classification.py
```

---

## Learning Outcomes

After completing this task, you will understand:

- Maximum margin classification
- Difference between linear and kernel SVM
- Impact of C and gamma
- Importance of feature scaling
- Cross-validation and hyperparameter tuning using GridSearchCV

---

## Future Improvements

- Apply SVM on real-world datasets
- Implement Pipeline (Scaling + SVM)
- Compare with Logistic Regression and Random Forest
- Perform advanced hyperparameter tuning
