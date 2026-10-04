**unsupervised-machine-learning**

**1. Project Overview**

This project focuses on applying unsupervised machine learning techniques and evaluating machine learning models using the Amazon Product Dataset.

K-Means and Hierarchical Clustering are used to identify groups of similar products. Principal Component Analysis (PCA) is used for dimensionality reduction and visualization.

Classification evaluation techniques are also demonstrated using Logistic Regression.

**2. Objectives**

- Understand K-Means clustering.
- Implement Hierarchical Clustering.
- Apply PCA for dimensionality reduction.
- Evaluate clustering using Silhouette Score.
- Perform cross-validation.
- Calculate Accuracy, Precision, Recall, and F1-score.
- Calculate ROC-AUC.
- Generate a Confusion Matrix.
- Perform hyperparameter tuning using GridSearchCV.

---

**3. Dataset**

 Amazon Product Dataset

The dataset contains Amazon product information such as:

- Product name
- Category
- Discounted price
- Actual price
- Discount percentage
- Product rating
- Rating count

**4. Features Used**

The following numerical features were selected for machine learning:

```text
discounted_price
actual_price
discount_percentage
rating_count
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

**5.Machine Learning Techniques**

1. K-Means Clustering

K-Means was used to divide the products into 3 clusters.

Silhouette Score:0.6372

2. Hierarchical Clustering

Agglomerative Hierarchical Clustering was performed with 3 clusters.

Silhouette Score:0.3880

3. PCA

Principal Component Analysis was used to reduce the four numerical features into two principal components.

The first two components explained approximately:

75.64% of the variance
Classification

A binary classification target was created from product ratings:

Rating >= 4  → Highly Rated (1)
Rating < 4   → Not Highly Rated (0)

Logistic Regression was used for classification evaluation.

**6.Model Evaluation**

The following metrics were used:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Results
Metric	Score
Accuracy	0.7585
Precision	0.7585
Recall	1.0000
F1-score	0.8627
ROC-AUC	0.6496
Cross-Validation

Five-fold cross-validation was performed using Logistic Regression.

Mean Cross-Validation Score:

0.7586
Hyperparameter Tuning

GridSearchCV was used to tune the Logistic Regression parameter C.

Parameter values tested:

0.01
0.1
1
10

Best parameter:

C = 0.01

Best Cross-Validation Score:

0.7586
Clustering Comparison
Algorithm	Clusters	Silhouette Score
K-Means	3	0.6372
Hierarchical Clustering	3	0.3880

Based on the Silhouette Score, K-Means produced better clustering results than Hierarchical Clustering for this experiment.

**7.requirements.txt**

pip install -r requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn

**8.Conclusion**

This project provided practical experience in unsupervised learning and model evaluation. K-Means and Hierarchical Clustering were implemented to identify product groups. PCA was used for dimensionality reduction and visualization. Silhouette Score was used to compare clustering performance.

Logistic Regression was also evaluated using cross-validation, confusion matrix, precision, recall, F1-score, and ROC-AUC. GridSearchCV was used for hyperparameter tuning.

The project improved practical understanding of clustering, dimensionality reduction, model evaluation, and hyperparameter optimization using Scikit-learn.
