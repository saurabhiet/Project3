
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

ad_data = pd.read_csv(
    "d:/adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
ad_data.tail()


# In[9]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import math
fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(ad_data.shape[1]) / cols)
for i, column in enumerate(ad_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if ad_data.dtypes[column] == np.object:
        ad_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        ad_data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()


# In[10]:


(ad_data["Country"].value_counts() / ad_data.shape[0]).head()


# In[13]:


def numencode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column].astype(str))
    return result, encoders

encoded_data, _ = numencode_features(ad_data)
sns.heatmap(encoded_data.corr(), square=True)
plt.show()


# In[15]:


del original_data["Education"]


# In[17]:


encoded_data, encoders = numencode_features(ad_data)
fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(encoded_data.shape[1]) / cols)
for i, column in enumerate(encoded_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    encoded_data[column].hist(axes=ax)
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()


# In[23]:


cols_to_use=encoded_data.columns.difference(['Target'])
#encoded_data[cols_to_use].tail()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(encoded_data[cols_to_use], encoded_data["Target"], train_size=0.70)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("float64"))


# In[30]:


from sklearn.metrics import f1_score
cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["Target"].classes_, yticklabels=encoders["Target"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score: %f" % f1_score(y_test, y_pred))
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort_values()
plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()

