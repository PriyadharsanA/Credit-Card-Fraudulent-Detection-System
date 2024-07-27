CREDIT CARD FRAUDULENT TRANSACTION DETECTION SYSTEM

#Importing the dataset
import pandas as pd
df1=pd.read_csv("C:/Users/User/Desktop/fraudTrain.csv")
df2=pd.read_csv("C:/Users/User/Desktop/fraudTest.csv")

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df2['gender']=le.fit_transform(df2['gender'])
df1['category']=le.fit_transform(df1['category'])
df2['category']=le.fit_transform(df2['category'])
df1['state']=le.fit_transform(df1['state'])
df2['state']=le.fit_transform(df2['state'])

#Taking only required attributes for classification
df1=df1[['lat','long','merch_lat','merch_long','amt','gender','category','state','is_fraud']]
df2=df2[['lat','long','merch_lat','merch_long','amt','gender','category','state','is_fraud']]

#Checking if the distribution of target is balanced or imbalanced
high_1=df1[df1.is_fraud==0]#Values=1289169 records
low_df1=df1[df1.is_fraud==1]#Value=7506 records
high_2=df2[df2.is_fraud==0]#Values=553574 records
low_df2=df2[df2.is_fraud==1]#Values=2145 records

#As it is highly imbalanced, we're going to underfit the records and create new datasets for training and testing respectively
high_df1=resample(high_1,replace=True,n_samples=7506)
new_df1=pd.concat([high_df1,low_df1])#Training set

high_df2=resample(high_2,replace=True,n_samples=2145)
new_df2=pd.concat([high_df2,low_df2])#Testing set

#Divide the set into x_train,y_train,x_test,y_test respectively
x_train=new_df1.drop(columns='is_fraud')
y_train=new_df1['is_fraud']
x_test=new_df2.drop(columns='is_fraud')
y_test=new_df2['is_fraud']

#Training and testing different models
#Importing required modules
from sklearn.metrics import confusion_matrix,classification_report

#1.RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train) #Model Training
result_1=rf.score(x_test,y_test)*100 #Model performance score
y_pred_1=rf.predict(x_test) #Model prediction
cm_1=confusion_matrix(y_test,y_pred_1) #Getting Confusion_matrix for checking accuracy of model
cr_1=classification_report(y_test,y_pred_1) #Classification_report for checking overall performance of model
print(result_1)
print(cm_1)
print(cr_1)
#Model results
'''
Result_1=93.1934731934732

Confusion_matrix:
[[2076   69]
 [ 223 1922]]
 
Classification_Report:        
              precision    recall  f1-score   support

           0       0.90      0.97      0.93      2145
           1       0.97      0.90      0.93      2145

    accuracy                           0.93      4290
   macro avg       0.93      0.93      0.93      4290
weighted avg       0.93      0.93      0.93      4290
'''
#2.Logistic Regression
from sklearn.liear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train) #Model Training
result_2=lr.score(x_test,y_test)*100 #Model performance score
y_pred_2=lr.predict(x_test) #Model prediction
cm_2=confusion_matrix(y_test,y_pred_2) #Getting Confusion_matrix for checking accuracy of model
cr_2=classification_report(y_test,y_pred_2) #Classification_report for checking overall performance of model
print(result_2)
print(cm_2)
print(cr_2)

#Model results
'''
Result_2=84.84848484848484
Confusion_matrix:
[[2021  124]
 [ 526 1619]]

 Classification_Report:
              precision    recall  f1-score   support

           0       0.79      0.94      0.86      2145
           1       0.93      0.75      0.83      2145

    accuracy                           0.85      4290
   macro avg       0.86      0.85      0.85      4290
weighted avg       0.86      0.85      0.85      4290

'''
#3.Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train) #Model Training
result_3=tree.score(x_test,y_test)*100 #Model performance score
y_pred_3=tree.predict(x_test) #Model prediction
cm_3=confusion_matrix(y_test,y_pred_3) #Getting Confusion_matrix for checking accuracy of model
cr_3=classification_report(y_test,y_pred_3) #Classification_report for checking overall performance of model
print(result_3)
print(cm_3)
print(cr_3)

#Model results
'''
Result_3=93.28671328671328
Confusion_matrix:
[[2024  121]
 [ 167 1978]]

 Classification_Report:
              precision    recall  f1-score   support

           0       0.92      0.94      0.93      2145
           1       0.94      0.92      0.93      2145

    accuracy                           0.93      4290
   macro avg       0.93      0.93      0.93      4290
weighted avg       0.93      0.93      0.93      4290
'''
#NOTE: THE SCORES, CONFUSION_MATRICES AND CLASSIFICATION_REPORTS MAY CHANGE WITH SUCCESSIVE RUNS BUT WILL MAINTAIN THE VALUE TO A CERTAIN RANGE
THANK YOU....
