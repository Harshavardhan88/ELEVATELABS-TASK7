# ELEVATELABS-TASK7
# Breast Cancer Classification with SVM

This task explores how Support Vector Machines (SVMs) can be used to classify breast cancer tumors as malignant or benign using a cleaned version of the Breast Cancer Wisconsin dataset. Both linear and non-linear classifiers are implemented and compared, with visualizations and hyperparameter tuning included.


## About the Dataset:

The dataset contains features computed from digitized images of fine needle aspirates (FNAs) of breast masses. Each row represents one tumor case, and each column is a computed feature (e.g., radius, texture, symmetry). The diagnosis column indicates whether the tumor is malignant (`M`) or benign (`B`).

- Input features: 30 numerical values
- Target: `diagnosis` (M = 1, B = 0 after encoding)
- Source: UCI Machine Learning Repository


##  What This Project Does:

1. **Data Loading & Preprocessing:**

   - Removes any nulls or duplicates
   - Drops unnecessary columns like IDs
   - Scales features using `StandardScaler`
   - Encodes diagnosis labels as binary

2. **SVM Training:**
     
     - Trains two models:
     - Linear SVM
     - RBF (Radial Basis Function) SVM

3. **Visualization:**

   - Reduces dimensions using PCA
   - Plots decision boundaries in 2D space

5. **Model Tuning:**

   - Uses `GridSearchCV` to find the best `C` and `gamma` values for the RBF kernel

7. **Evaluation:**

   - Accuracy score on test data
   - Full classification report (precision, recall, F1-score)
   - 5-fold cross-validation to check model consistency



## Libraries Used:

- `pandas`, `numpy` for data handling
- `matplotlib`, `seaborn` for plots
- `scikit-learn` for model training, tuning, and evaluation

##  Example Output:

- Accuracy with RBF SVM: ~98%
- Best Parameters after Grid Search: e.g., `C=10`, `gamma=0.01`
- Clear decision boundary plots
- Detailed metrics showing the model handles both classes well

##  Why SVM?

SVMs are powerful for binary classification tasks, especially when:
- The data is not linearly separable (hence the RBF kernel)
- You're working with high-dimensional data
- You want a model that balances margin size and classification error

This task shows how to apply SVMs practically, tune them properly, and interpret their results.

