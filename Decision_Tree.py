#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
sns.set()

from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.metrics import plot_roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# In[2]:


#Loading Data
df=pd.read_csv("dataDT.csv")
df.head()


# In[3]:


df.info()


# In[4]:


#Summary Statistics
df.describe()


# In[5]:


df.columns
#Excluding first column
df=df.iloc[:,1:]
df


# In[6]:


#Checking data types
df.dtypes


# In[7]:


#Checking for missing values
df.isna().sum()


# In[8]:


#Checking for duplicates
duplicate = df[df.duplicated()]
print("Duplicate rows: \n",duplicate)


# In[9]:


#Dropping Duplicates
df=df.drop_duplicates(ignore_index=True)


# In[10]:


#Checking if data is imbalanced
df.target.value_counts()


# In[11]:


## EDA ##


# In[12]:


#Barplot for fasting blood sugar
df['fbs'].value_counts().plot.bar()
plt.xlabel('fbs')
plt.ylabel('Count')
plt.title('fbs')


# In[13]:


#Barplot for exercise induced angina
df['exang'].value_counts().plot.bar()
plt.xlabel('exang')
plt.ylabel('Count')
plt.title('exang')


# In[14]:


#Histogram for resting blood pressure
plt.hist(df['trestbps'])
plt.xlabel('trestbps')
plt.ylabel('Count')
plt.title('trestbps ')


# In[15]:


#Histogram for chlolesterol level
plt.hist(df['chol'])
plt.xlabel('chol')
plt.ylabel('Count')
plt.title('Chol')


# In[16]:


#Sex-wise distribution of Heart disease
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.sex,df.target).plot(kind='bar')
plt.title('Target v/s Sex ')
plt.xlabel('sex')
plt.ylabel('Frequency')


# In[17]:


#Distribution of chest pain type and heart disease
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.cp,df.target).plot(kind='bar')
plt.title('Target v/s cp ')
plt.xlabel('cp')
plt.ylabel('Frequency')


# In[18]:


## Encoding Categorical Variables ##


# In[19]:


le=LabelEncoder()
df.sex=le.fit_transform(df.sex)
df.fbs=le.fit_transform(df.fbs)
df.exang=le.fit_transform(df.exang)

df=pd.get_dummies(df,drop_first=True)

x=df.drop(['target'],axis=1)
y= df.iloc[:,-4]


# In[20]:


#Splitting into Train and test
x_train, x_test, y_train, y_test = train_test_split( x, y,test_size=0.20, random_state=10)


# In[21]:


## Fitting Model ##


# In[22]:


# Create Decision Tree classifer object
dt= DecisionTreeClassifier(random_state=5,max_depth=10,min_samples_leaf=5)
# Train Decision Tree Classifer
dt = dt.fit(x_train,y_train)
y_train_pred = dt.predict(x_train)
#Predict the response for test dataset
y_pred = dt.predict(x_test)

print("Accuracy Train:",metrics.accuracy_score(y_train, y_train_pred))
print("Accuracy Test:",metrics.accuracy_score(y_test, y_pred))


# In[23]:


#Confusion Matrix
CM=confusion_matrix(y_test,y_pred)
print("confusion matrix:\n ",CM)

#plot
sns.heatmap(CM, annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[24]:


#ROC Curve
plot_roc_curve(dt,x_test,y_test)


# In[ ]:




