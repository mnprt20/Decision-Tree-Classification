# Decision-Tree-Regression
**Diabetes Prediction using Decision Tree Classifier**
=====================================================

### Overview
This project uses a Decision Tree Classifier to predict the outcome of diabetes based on various patient attributes. The dataset used is the Pima Indians Diabetes Database, which contains 768 samples and 9 features.

### Dataset
The dataset is stored in a CSV file named "diabetes.csv" and is located at "C:\\infosys\D\diabetes.csv". The dataset contains the following features:

* **Pregnancies**
* **Glucose**
* **BloodPressure**
* **SkinThickness**
* **Insulin**
* **BMI**
* **DiabetesPedigreeFunction**
* **Age**
* **Outcome** (target attribute)

### Code Explanation
#### Importing Libraries
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```
####  Data Loading and Exploration
```python
df = pd.read_csv("C:\\infosys\D\diabetes.csv")
print(df)
```
#### Correlation Matrix
```python
corr_matrix = df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', square=True)
plt.title('Correlation Matrix')
plt.show()
```
#### Pairplot
```python
sns.pairplot(df, vars=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
plt.show()
```
#### Handling Missing Values
```python
print(df.isnull().values.any())
```
#### Feature and Target Attribute Selection
```python
target_attribute = 'Outcome'
X = df.drop(columns=[target_attribute])
y = df[target_attribute]
```
#### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
#### Decision Tree Classifier
```python
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
```
#### Evaluation Metrics
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
```
#### Visualization
```python
sns.pairplot(df, x_vars=X.columns, y_vars=[target_attribute], height=3, aspect=1)
plt.show()
plt.figure(figsize=(10,8))
plot_tree(dtc, feature_names=X.columns, filled=True)
plt.show()
plt.figure(figsize=(15,12))
plot_tree(dtc, 
          feature_names=X.columns, 
          filled=True, 
          rounded=True,
          fontsize=10, 
          class_names=['0', '1'],
          impurity=False, 
          node_ids=True,  
          max_depth=None) 
plt.show()
```
#### Requirements
* **Python 3.x**
* **pandas**
* **seaborn**
* **matplotlib**
* **scikit-learn**
#### Running the Code
To run the code, simply execute the Python script in a suitable environment. The code will load the dataset, perform the necessary data exploration and visualization, train the Decision Tree Classifier, and evaluate its performance.
