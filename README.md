# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3.Split the dataset using train_test_split.
4.Calculate Y_Pred and accuracy.
5.Print all the outputs.
6.End the Program.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRADEEP E
RegisterNumber: 212223230149
*/
```
```
 import pandas as pd
 data=pd.read_csv("/content/spam.csv",encoding='Windows=1252')
 data.head()
```
![Screenshot 2024-11-10 163201](https://github.com/user-attachments/assets/0f83bc5e-a5a0-4cbe-9f7b-a4d48a53bdc3)

```
data.tail()
```
![Screenshot 2024-11-10 163210](https://github.com/user-attachments/assets/38333906-2e84-42a8-96f1-8b5871fced6b)

```
data.info()
```
![Screenshot 2024-11-10 163216](https://github.com/user-attachments/assets/958ba98a-bfc3-4007-aa9a-8d262d8e211a)

```
data.isnull().sum()
```

![Screenshot 2024-11-10 163222](https://github.com/user-attachments/assets/97312c59-a769-403a-be86-7422cfce9c81)

```
x=data["v2"].values
y=data["v1"].values
y.shape
```
![Screenshot 2024-11-10 163227](https://github.com/user-attachments/assets/56e57b6a-8024-439b-be7d-cca68129a976)

```
x.shape
```
![Screenshot 2024-11-10 163232](https://github.com/user-attachments/assets/d561b5ec-9071-4db4-8d96-461436764bd3)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```
![Screenshot 2024-11-10 163237](https://github.com/user-attachments/assets/725693d2-4572-4039-995e-0c85fde0c5d9)

```
y_train.shape
```
![Screenshot 2024-11-10 163242](https://github.com/user-attachments/assets/0a9a7f53-429f-46ab-be7f-7def4efbedcf)

```
x_test.shape
```
![Screenshot 2024-11-10 163249](https://github.com/user-attachments/assets/34522de6-a89c-4bce-8ccb-a0ed4bc3f281)

```
y_test.shape
```
![Screenshot 2024-11-10 163253](https://github.com/user-attachments/assets/8d2f1da3-83a9-40b0-86a8-47c8706c8eb0)

```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.fit_transform(x_test)
x_train.shape
```
![Screenshot 2024-11-10 163308](https://github.com/user-attachments/assets/d9b0261b-6bdc-4ac0-8359-df0d5e0c5aaf)

```
type(x_train)
```
![Screenshot 2024-11-10 163315](https://github.com/user-attachments/assets/21ce9105-f8e4-4b95-9d8a-d84ffd5b11b1)

```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
![Screenshot 2024-11-10 163322](https://github.com/user-attachments/assets/18c6e3e7-6357-4fe8-9348-40780f69fcd8)

```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![Screenshot 2024-11-10 163334](https://github.com/user-attachments/assets/5650a311-2c83-4743-a811-0a227e7c7cc4)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
