
# coding: utf-8

# In[39]:

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score


# In[4]:

#importing the dataset
dataset = pd.read_csv("C:\\Users\\narayanreddy\\Desktop\\python\\My Projects\\Kaggle credit card default payments\\UCI_Credit_Card.csv")


# In[5]:

#Renaming the dependent variable as default
dataset = dataset.rename(columns = {'default.payment.next.month':'default'})


# In[6]:

print(dataset.columns)


# In[7]:

#checking for missing values in the dataset
print(dataset.isnull().sum())


# In[8]:

#Exploratory Analysis
sns.countplot(x='SEX', data = dataset)


# In[9]:

sns.countplot(x='EDUCATION', data = dataset)


# In[10]:

sns.countplot(x='MARRIAGE', data = dataset)


# In[11]:

sns.countplot(x='AGE', data = dataset)


# In[12]:

dataset.groupby('SEX')['SEX'].count()


# In[13]:

dataset.groupby('EDUCATION')['EDUCATION'].count()


# In[14]:

dataset.groupby('MARRIAGE')['MARRIAGE'].count()


# In[15]:

dataset.groupby('AGE')['AGE'].count()


# In[16]:

#Marriage, merging 0 to 3
dataset['MARRIAGE'] = dataset['MARRIAGE'].map({0:3, 1:1, 2:2, 3:3})
MARRIAGE_count= dataset.groupby('MARRIAGE').MARRIAGE.count()
print(MARRIAGE_count)


# In[17]:

sns.countplot(x='MARRIAGE', data = dataset)


# In[18]:

#Education is ordinal: merging 0,5,6 to 4
dataset['EDUCATION'] = dataset['EDUCATION'].map({0:4, 1:1, 2:2, 3:3, 4:4, 
    5:4, 6: 4})
EDUCATION_count= dataset.groupby('EDUCATION').EDUCATION.count()
print(EDUCATION_count)


# In[19]:

sns.countplot(x='EDUCATION', data = dataset)


# In[20]:

#Repayments: -1: pay duly; 1: payment delayed by 1 month; 2: payment delayed by 2 months and so....

sns.countplot(x='PAY_0', data = dataset)


# In[22]:

sns.countplot(x='PAY_2', data = dataset)


# In[23]:

sns.countplot(x='PAY_3', data = dataset)


# In[24]:

sns.countplot(x='PAY_4', data = dataset)


# In[25]:

sns.countplot(x='PAY_5', data = dataset)


# In[27]:

sns.countplot(x='PAY_6', data = dataset)


# In[28]:

#-1,-2 & 0 indicates same regards with repayment status and hence we can merge all of them to 0
dataset["PAY_0"] = dataset["PAY_0"].map({-1:0,-2:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})
dataset["PAY_2"] = dataset["PAY_2"].map({-1:0,-2:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})
dataset["PAY_3"] = dataset["PAY_3"].map({-1:0,-2:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})
dataset["PAY_4"] = dataset["PAY_4"].map({-1:0,-2:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})
dataset["PAY_5"] = dataset["PAY_5"].map({-1:0,-2:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})
dataset["PAY_6"] = dataset["PAY_6"].map({-1:0,-2:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8})


# In[29]:

#check for default payments for next month
default_count= dataset.groupby('default').default.count()
print(default_count)


# In[30]:

sns.countplot(x='default', data = dataset)


# In[31]:

#Gender affecting the default payment
pd.crosstab(dataset['default'],dataset['SEX'])


# In[32]:

#Education affecting the default payment
pd.crosstab(dataset['default'],dataset['EDUCATION'])


# In[33]:

#Marital status affecting the default payment
pd.crosstab(dataset['default'],dataset['MARRIAGE'])


# In[34]:

#correlation plot
plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(), annot =True)


# In[35]:

plt.figure(figsize=(20,20))
ax=sns.boxplot(data=dataset,orient="h",palette="Set2")


# In[36]:

#Splitting the dataset
dataset_train,dataset_test = dataset.iloc[:,1:len(dataset.columns)-1],dataset.iloc[:,len(dataset.columns)-1]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_train, dataset_test, test_size = 0.2, random_state = 0)


# In[37]:

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[40]:

#Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[41]:

accuracy_score(y_test, y_pred)


# In[42]:

precision_score(y_test, y_pred)


# In[43]:

recall_score(y_test, y_pred)


# In[44]:

import statsmodels.api as sm
from scipy import stats
logit_model = sm.Logit(y_train, sm.add_constant(X_train)).fit()


# In[45]:

print (logit_model.summary())


# In[46]:

#Removing high Pvalue variables
X_trainnew=X_train.drop(['BILL_AMT2','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT3','PAY_AMT4','PAY_AMT5'],axis=1)
X_testnew=X_test.drop(['BILL_AMT2','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT3','PAY_AMT4','PAY_AMT5'],axis=1)


# In[47]:

#Logestic Regression with seleted variables
classifier1 = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
classifier1.fit(X_trainnew, y_train)
y_prednew = classifier1.predict(X_testnew)


# In[48]:

newcm=confusion_matrix(y_test,y_prednew)
print(newcm)


# In[49]:

accuracy_score(y_test, y_prednew)


# In[50]:

precision_score(y_test, y_prednew)


# In[51]:

recall_score(y_test, y_prednew)


# In[52]:

logit_modelnew = sm.Logit(y_train, sm.add_constant(X_trainnew)).fit()
print (logit_modelnew.summary())


# In[53]:

# Random forest method with seleted variables
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=50,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=600, n_jobs=-1, oob_score=False,
            random_state=None, verbose=0, warm_start=False)
forest.fit(X_trainnew, y_train)
y_pred2 = forest.predict(X_testnew)


# In[54]:

cm1=confusion_matrix(y_test,y_pred2)
print(cm1)


# In[55]:

accuracy_score(y_test, y_pred2)


# In[56]:

precision_score(y_test, y_pred2)


# In[57]:

recall_score(y_test, y_pred2)


# In[58]:

from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred2)


# In[59]:

roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


# In[60]:

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[62]:

# Applying k-Fold Cross Validation on RF model
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = forest, X = X_trainnew, y = y_train, cv = 10)
accuracies.mean()


# In[64]:

# Applying kNN(k- nearest neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_trainnew, y_train)
knn_score_train = knn.score(X_trainnew, y_train)
print("Training score: ",knn_score_train)
knn_score_test = knn.score(X_testnew, y_test)
print("Testing score: ",knn_score_test)


# In[ ]:



