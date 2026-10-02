# Comparison of Different Classification Algorithms

## 📌 Overview

This project implements and compares multiple **classification algorithms** using a suitable classification dataset.

The main purpose of this experiment is to understand how different machine learning classification algorithms perform on the same dataset and to compare their performance using standard evaluation metrics.

The following classification algorithms are implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

The **Iris dataset** is used for this experiment. Each algorithm is trained and evaluated using the same training and testing data so that their performance can be compared fairly.

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

---

## 🎯 Aim

To compare different **classification algorithms** using a suitable dataset and evaluate their performance using **Accuracy, Precision, Recall, and Confusion Matrix**.

---

## 🎯 Objectives

The main objectives of this experiment are:

1. To understand the concept of classification in machine learning.
2. To load and explore a suitable classification dataset.
3. To perform basic data preprocessing and analysis.
4. To separate features and target variables.
5. To divide the dataset into training and testing sets.
6. To apply feature scaling where required.
7. To implement multiple classification algorithms.
8. To train the classification models using training data.
9. To generate predictions using the trained models.
10. To evaluate each model using Accuracy, Precision, and Recall.
11. To generate confusion matrices for the classification models.
12. To compare the performance of different classification algorithms.

---

## 🧠 Theory

### What is Classification?

**Classification** is a supervised machine learning technique in which a model learns from labelled training data and predicts the class or category of new observations.

For example:

- Email → Spam / Not Spam
- Patient → Disease / No Disease
- Transaction → Fraud / Genuine
- Flower → Setosa / Versicolor / Virginica

The output of a classification model is a **discrete class label**.

---

## 📊 Classification in Machine Learning

Classification belongs to **supervised learning** because the training dataset contains both:

- Input features
- Known target labels

The model learns the relationship between the input features and the target classes and uses this learned relationship to predict the class of unseen observations.

---

## 📚 Dataset Used

The **Iris dataset** is used in this experiment.

The Iris dataset contains measurements of iris flowers and is commonly used for demonstrating classification algorithms.

### Features

The dataset contains four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes

The target variable contains three classes:

- Setosa
- Versicolor
- Virginica

Therefore, this is a **multi-class classification problem**.

---

## 🔄 Machine Learning Workflow

The complete machine learning workflow used in this experiment is:

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Prediction
   ↓
Performance Evaluation
   ↓
Algorithm Comparison
```

---

# 🤖 Classification Algorithms

## 1. Logistic Regression

**Logistic Regression** is a supervised classification algorithm that predicts the probability of a class using a logistic function.

Although its name contains "Regression", it is widely used for classification problems.

For multi-class classification, Logistic Regression can model the probability of multiple classes.

The implementation uses:

```python
from sklearn.linear_model import LogisticRegression
```

The model used in this experiment is:

```python
LogisticRegression(max_iter=500)
```

### Characteristics

- Simple and interpretable
- Effective for many classification problems
- Works well with scaled numerical features
- Provides probability-based predictions

---

## 2. Decision Tree

A **Decision Tree** is a tree-based supervised learning algorithm.

It makes predictions by repeatedly splitting the data based on feature conditions.

The basic structure contains:

```text
Root Node
    ↓
Decision Nodes
    ↓
Branches
    ↓
Leaf Nodes
```

The implementation uses:

```python
from sklearn.tree import DecisionTreeClassifier
```

The model used in this experiment is:

```python
DecisionTreeClassifier(random_state=42, max_depth=5)
```

### Characteristics

- Easy to understand
- Can model non-linear relationships
- Does not require feature scaling
- Can be visualized as a decision tree

---

## 3. Random Forest

**Random Forest** is an ensemble learning algorithm that combines multiple decision trees.

Each tree generates a prediction, and the forest combines the individual predictions to produce the final classification.

The implementation uses:

```python
from sklearn.ensemble import RandomForestClassifier
```

The model used in this experiment is:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=1
)
```

### Characteristics

- Ensemble of multiple decision trees
- Robust to many types of data variation
- Can model complex relationships
- Does not require feature scaling
- Can provide feature importance information

---

## 4. Support Vector Machine (SVM)

**Support Vector Machine (SVM)** is a supervised learning algorithm that attempts to find an optimal decision boundary between classes.

For non-linear classification problems, kernels can be used to transform the feature space.

In this experiment, the **RBF kernel** is used.

The implementation uses:

```python
from sklearn.svm import SVC
```

The model used is:

```python
SVC(
    kernel="rbf",
    C=1.0,
    random_state=42
)
```

### Characteristics

- Effective for many classification problems
- Can model non-linear boundaries using kernels
- Sensitive to feature scaling
- Uses support vectors to define the decision boundary

---

## 5. K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a distance-based classification algorithm.

It predicts the class of a new observation by examining the classes of its nearest neighboring observations.

The implementation uses:

```python
from sklearn.neighbors import KNeighborsClassifier
```

The model used is:

```python
KNeighborsClassifier(n_neighbors=5)
```

The value:

```text
K = 5
```

means that the five nearest observations are considered when making a prediction.

### Characteristics

- Simple and easy to understand
- Distance-based algorithm
- Sensitive to feature scaling
- Does not require an explicit model-building phase
- Performance depends on the choice of `K`

---

## 6. Gaussian Naive Bayes

**Gaussian Naive Bayes** is a probabilistic classification algorithm based on Bayes' theorem.

Bayes' theorem is:

```text
P(A|B) = P(B|A)P(A) / P(B)
```

Gaussian Naive Bayes assumes that continuous features follow a Gaussian or normal distribution within each class.

The implementation uses:

```python
from sklearn.naive_bayes import GaussianNB
```

The model is created using:

```python
GaussianNB()
```

### Characteristics

- Simple and computationally efficient
- Probabilistic approach
- Works well with numerical features
- Based on the assumption of conditional independence between features

---

# 📈 Model Evaluation

After training all models, predictions are generated using the test dataset.

The models are compared using several evaluation metrics.

---

## 1. Accuracy

Accuracy represents the proportion of correctly classified observations out of all observations.

```text
Accuracy = Correct Predictions / Total Predictions
```

A higher accuracy indicates that a larger proportion of test observations were classified correctly.

---

## 2. Precision

Precision measures how many observations predicted as a particular class are actually members of that class.

For binary classification:

```text
Precision = TP / (TP + FP)
```

where:

- **TP** = True Positive
- **FP** = False Positive

For this multi-class experiment, **macro averaging** is used so that each class contributes equally to the overall metric.

---

## 3. Recall

Recall measures how many actual observations belonging to a class were correctly identified.

For binary classification:

```text
Recall = TP / (TP + FN)
```

where:

- **TP** = True Positive
- **FN** = False Negative

Macro averaging is used for the multi-class evaluation.

---

## 4. Confusion Matrix

A confusion matrix provides a detailed view of actual and predicted class labels.

For the Iris dataset, the confusion matrix contains three classes:

```text
                 Predicted
              Setosa  Versicolor  Virginica

Actual
Setosa           ...
Versicolor       ...
Virginica        ...
```

The diagonal elements represent correctly classified observations.

The off-diagonal elements represent misclassifications.

---

## 5. Classification Report

A classification report provides important evaluation metrics for each class, including:

- Precision
- Recall
- F1-score
- Support

It provides a detailed view of the model's classification performance.

---

# 📊 Model Comparison

The performance of all six algorithms is collected into a comparison table containing:

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Decision Tree | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Random Forest | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| SVM | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| KNN | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Gaussian Naive Bayes | Calculated in notebook | Calculated in notebook | Calculated in notebook |

The exact metric values are generated when the Jupyter Notebook is executed.

---

## 📊 Visualization

Confusion matrices are generated for the classification models to visualize correct and incorrect predictions.

A final comparison chart is also generated to compare the evaluation metrics of the different algorithms.

The comparison includes:

- Accuracy
- Precision
- Recall

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Experiment development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Machine learning algorithms and evaluation |

---

# 📂 Project Structure

```text
ML_Comparison_of_Different_Classification_Algorithms/
│
├── Comparison_of_Different_Classification_Algorithms.ipynb
├── .gitignore
└── README.md
```

---

# 💻 Requirements

Install the required Python libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

If you are using the Anaconda environment:

```bash
conda activate ml
```

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# ▶️ How to Run

## Step 1: Clone the Repository

```bash
git clone https://github.com/SushantVasagade/ML_Comparison_of_Different_Classification_Algorithms.git
```

## Step 2: Open the Project

```bash
cd ML_Comparison_of_Different_Classification_Algorithms
```

## Step 3: Activate the Environment

```bash
conda activate ml
```

## Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

## Step 5: Run the Notebook

Open:

```text
Comparison_of_Different_Classification_Algorithms.ipynb
```

and execute the cells sequentially.

---

# 📌 Key Learning Outcomes

After completing this experiment, the following concepts are understood:

- Supervised learning
- Classification
- Multi-class classification
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Train-test splitting
- Feature scaling
- Model training
- Model prediction
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Model comparison

---

# 🔍 Algorithm Comparison

| Algorithm | Main Concept | Feature Scaling |
|---|---|---|
| Logistic Regression | Probability-based classification | Required |
| Decision Tree | Rule-based tree splitting | Not required |
| Random Forest | Ensemble of decision trees | Not required |
| SVM | Maximum-margin classification | Required |
| KNN | Distance-based classification | Required |
| Gaussian Naive Bayes | Probabilistic classification | Generally not required |

---

# ⚠️ Important Considerations

Model performance depends on several factors, including:

- Dataset characteristics
- Feature quality
- Data preprocessing
- Feature scaling
- Hyperparameter selection
- Training and testing split
- Evaluation metric

Therefore, classification algorithms should be evaluated using appropriate metrics rather than relying on only one measure.

---

# 🌍 Real-World Applications

Classification algorithms are widely used in:

- Medical diagnosis support
- Email spam detection
- Fraud detection
- Customer classification
- Sentiment analysis
- Image classification
- Credit risk classification
- Document classification
- Disease classification
- Recommendation systems

---

# 📝 Conclusion

Different classification algorithms were implemented and compared using the Iris dataset.

The experiment demonstrated the complete classification workflow, including data exploration, preprocessing, feature-target separation, train-test splitting, feature scaling, model training, prediction, and performance evaluation.

Six classification algorithms were studied:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. K-Nearest Neighbors
6. Gaussian Naive Bayes

The models were evaluated using **Accuracy, Precision, Recall, Confusion Matrix, and Classification Report**.

The experiment demonstrates that different algorithms can produce different results on the same dataset and that model evaluation should consider appropriate performance metrics rather than relying on a single measure.

---

# 👨‍💻 Author

**Sushant Vasagade**

B.Tech – Information Technology

Government College of Engineering, Karad

---

# 📚 Repository

[![GitHub Repository](https://img.shields.io/badge/GitHub-Classification_Algorithms-black?logo=github)](https://github.com/SushantVasagade/ML_Comparison_of_Different_Classification_Algorithms)

**Repository:** [ML_Comparison_of_Different_Classification_Algorithms](https://github.com/SushantVasagade/ML_Comparison_of_Different_Classification_Algorithms)

---

# 📚 Machine Learning Lab

**Experiment:** Comparison of Different Classification Algorithms

**Domain:** Machine Learning

**Dataset:** Iris Dataset

**Algorithms:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gaussian Naive Bayes
