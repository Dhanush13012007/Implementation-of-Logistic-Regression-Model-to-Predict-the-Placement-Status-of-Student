# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

 
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.Dhanush
RegisterNumber:  25009955
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()data1.isnull().sum()

data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="1275" height="219" alt="Screenshot 2025-12-12 132808" src="https://github.com/user-attachments/assets/04c3aa2d-7e96-4aed-8950-06d70e93f2b2" />

<img width="1111" height="214" alt="Screenshot 2025-12-12 132819" src="https://github.com/user-attachments/assets/f1f67a37-1bc5-4ce9-84c8-851625910385" />


<img width="399" height="305" alt="Screenshot 2025-12-12 132832" src="https://github.com/user-attachments/assets/fb45cae7-4c78-45bb-baf6-526fed48dd46" />


<img width="121" height="45" alt="Screenshot 2025-12-12 132840" src="https://github.com/user-attachments/assets/144394d7-f4b1-4155-bb10-077506d41352" />


<img width="1088" height="439" alt="Screenshot 2025-12-12 132851" src="https://github.com/user-attachments/assets/afebf0dd-a42c-4728-876c-cf28a4c84c7d" />


<img width="532" height="260" alt="Screenshot 2025-12-12 132900" src="https://github.com/user-attachments/assets/58ebb1f1-bff3-47c0-99e3-27cfdcde83c3" />


<img width="827" height="67" alt="Screenshot 2025-12-12 132909" src="https://github.com/user-attachments/assets/eae0c8e6-5180-4442-942b-b556ee6b06df" />

<img width="275" height="24" alt="Screenshot 2025-12-12 132915" src="https://github.com/user-attachments/assets/dda7d2a2-ea94-45a3-adb4-9e39865dd4fe" />

<img width="418" height="62" alt="Screenshot 2025-12-12 132922" src="https://github.com/user-attachments/assets/ef24924c-c16c-4cf4-aa7a-df5517449697" />


<img width="686" height="198" alt="Screenshot 2025-12-12 132933" src="https://github.com/user-attachments/assets/0a8aa57e-1c1c-43c4-ae85-959e9bbeda70" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
