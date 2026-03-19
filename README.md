# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy
5.Generate Confusion Matrix
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: vishnuram g n
RegisterNumber:  212225240187
*/

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 

iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head())

<img width="310" height="40" alt="Screenshot 2026-03-19 142050" src="https://github.com/user-attachments/assets/bc8af1ef-70d5-4d0f-a069-6ab999daa1dd" />

X = df.drop('target',axis=1) 
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)

y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

<img width="1020" height="377" alt="Screenshot 2026-03-19 142157" src="https://github.com/user-attachments/assets/6953f3a0-0f97-432a-9be1-cdbedeb71f09" />

cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)

<img width="324" height="152" alt="Screenshot 2026-03-19 142244" src="https://github.com/user-attachments/assets/95c5c361-bf9b-40ab-b86f-69954dafa3c9" />

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
<img width="916" height="653" alt="Screenshot 2026-03-19 142319" src="https://github.com/user-attachments/assets/40ed3e27-0f59-43bb-a35e-0c7562518b7e" />

```

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
