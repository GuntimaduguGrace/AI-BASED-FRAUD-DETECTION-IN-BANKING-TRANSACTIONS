#!/usr/bin/env python
# coding: utf-8

# Online payment is the most popular transaction method in the world today. However, with an increase in online payments also comes a rise in payment fraud. The objective of this study is **to identify fraudulent and non-fraudulent payments**. The dataset is collected from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments. 
# 
# 
# The dataset consists of 10 variables:
# * **step**: represents a unit of time where 1 step equals 1 hour
# * **type**: type of online transaction
# * **amount**: the amount of the transaction
# * **nameOrig**: customer starting the transaction
# * **oldbalanceOrg**: balance before the transaction
# * **newbalanceOrig**: balance after the transaction
# * **nameDest**: recipient of the transaction
# * **oldbalanceDest**: initial balance of recipient before the transaction
# * **newbalanceDest**: the new balance of recipient after the transaction
# * **isFraud**: fraud transaction

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy as sp
from tabulate import tabulate
import random
import tensorflow as tf

import os
for dirname, _, filenames in os.walk("C:/Users/gulig/Desktop/Projects2k25"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # <div style='background:darkblue'>  <font color='white'> <b>&nbsp;Exploratory Data Analysis</b></font> </div>

# In[2]:


df = pd.read_csv("C:/Users/gulig/Desktop/Projects2k25/onlinepayment/onlinefraud.csv")

df.drop('isFlaggedFraud', axis=1, inplace=True)

df.info()


# * The dataset consists of 6,362,620 observations.

# In[3]:


df.sample(5)


# In[4]:


df.isnull().sum()


# * There are no missing values.

# In[5]:


fraud_min_max = [
    ['amount', df.amount.min(), df.amount.max()],
    ['oldbalanceOrg', df.oldbalanceOrg.min(), df.oldbalanceOrg.max()],
    ['newbalanceOrig', df.newbalanceOrig.min(), df.newbalanceOrig.max()],
    ['oldbalanceDest', df.oldbalanceDest.min(), df.oldbalanceDest.max()],
    ['isFraud', df.isFraud.min(), df.isFraud.max()]
] 

print(
    tabulate(
        fraud_min_max,
        headers=['columns', 'min value', 'max value'],
        showindex=True,
        tablefmt='github',
        numalign='right'
    )
)


# In[6]:


# Downcast numerical columns with smaller dtype
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = pd.to_numeric(df[col], downcast='float')
    if df[col].dtype == 'int64':
        df[col] = pd.to_numeric(df[col], downcast='unsigned')

# Use category dtype for categorical column
df['type'] = df['type'].astype('category')


# In[7]:


# Check duplicate values
df.duplicated().sum()


# * There are no duplicate values.

# In[8]:


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8,6)


# <h2>
#     <div style='background:steelblue'> 
#     <font color='white'>
#     <b>&nbsp; Univariate data visualization</b>
#     </font>
#     </div>
# </h2>

# In[9]:


df['step'].value_counts()


# * There are **743** steps, and every step has **at least 2** occurrences.

# In[10]:


ax = sns.countplot(x='type', data=df, palette='PuBu')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Count plot of transaction type')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.ylabel('Number of transactions')


# * **Cash out** is **the most numerous** transaction type, followed by payment, cash in, transfer and debit types.

# In[11]:


sns.kdeplot(df['amount'], linewidth=4)
plt.title('Distribution of transaction amount')


# * The distribution of transaction amounts is **right skewed**. 
# * This indicates that most values are clustered around the left tail of the distribution, with the longer right tail. 
# * (mode < median < mean)

# In[12]:


df['nameOrig'].value_counts()


# * There are **6353307** initial customers, and every step has **at least 1** occurrence.

# In[13]:


sns.kdeplot(df['oldbalanceOrg'], linewidth=4)
plt.title('Distribution of transaction amount')


# * The distribution of pre-transaction balances of the initial customers is **right skewed**.

# In[14]:


sns.kdeplot(df['newbalanceOrig'], linewidth=4)
plt.title('Distribution of transaction amount')


# * The distribution of post-transaction balances of the initial customers is **right skewed**.

# In[15]:


df['nameDest'].value_counts()


# * There are **2722362** recipients, and every step has **at least 1** occurrence.

# In[16]:


sns.kdeplot(df['oldbalanceDest'], linewidth=4)
plt.title('Distribution of transaction amount')


# * The distribution of pre-transaction balances of the recipient is **right skewed**.

# In[17]:


sns.kdeplot(df['newbalanceDest'], linewidth=4)
plt.title('Distribution of transaction amount')


# * The distribution of post-transaction balances of the recipient is **right skewed**.

# In[18]:


ax = sns.countplot(x='isFraud', data=df, palette='PuBu')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Count plot of fraud transaction')
plt.ylabel('Number of transactions')

del ax


# * There are much **more non-fraudulent transactions** than fraudulent transactions.

# <h2>
#     <div style='background:steelblue'> 
#     <font color='white'>
#     <b>&nbsp; Bivariate data visualization</b>
#     </font>
#     </div>
# </h2>

# In[19]:


fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(x='type', data=df, hue='isFraud', palette='PuBu', ax=ax[0])
for container in ax[0].containers:
    ax[0].bar_label(container)
ax[0].set_title('Count plot of transaction type')
ax[0].legend(loc='best')
ax[0].set_ylabel('Number of transactions')

df2 = df.groupby(['type', 'isFraud']).size().unstack()
df2.apply(lambda x : round(x/sum(x)*100, 2), axis=1).plot(kind='barh', stacked=True, color=['lightsteelblue', 'steelblue'], ax=ax[1])
for container in ax[1].containers:
    ax[1].bar_label(container, label_type='center')
ax[1].set_title('Count plot of transaction type')
ax[1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
ax[1].set_ylabel('Number of transactions')
ax[1].grid(axis='y')


# * Fraudulent transactions only occur in debit and transfer types.

# In[20]:


df['quantity'] = pd.cut(df['amount'], 5, labels=['very low', 'low', 'moderate', 'high', 'very high'])

ax = sns.countplot(x='quantity', data=df, hue='isFraud', palette='PuBu')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Count plot of amount quantity')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.ylabel('Number of transactions')


# * All fraudulent transactions fall into the category of very low amounts. 
# * This suggests that in most cases, small transactions are more prone to fraudulent transactions.

# In[21]:


df1 = df[df['isFraud']==1]
df2 = df1['step'].value_counts().head(10)
ax = df2.plot(kind='bar', color='lightsteelblue')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Top 10 steps that often lead to fraudulent transactions')
plt.ylabel('Number of fraudulent transactions')
plt.xlabel('Step')
plt.grid(axis='x')

del ax, df2


# * **Step 212** has the highest number of fraudulent transactions, 40 cases. 
# * This indicates that Step 212 is the step that will most likely lead to fraudulent transactions.

# In[22]:


df['oldbalanceOrg_amt'] = pd.cut(df['oldbalanceOrg'], 5, labels=['very low', 'low', 'moderate', 'high', 'very high'])

ax = sns.countplot(x='oldbalanceOrg_amt', data=df, hue='isFraud', palette='PuBu')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Count plot of initial customers pre-transaction balance amount')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.ylabel('Number of transactions')
plt.xlabel('Initial customers pre-transaction balance amount')


# * Initial customers with **very low pre-transaction balances** has the highest number of fraudulent transactions. 
# * This means that initial customers with very low pre-transaction balances may be more likely to fall for a fraudulent transaction.

# In[23]:


df2 = df1['nameOrig'].value_counts().head(10)
ax = df2.plot(kind='bar', color='lightsteelblue')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Top 10 initial customers fall for a fraudulent transactions')
plt.ylabel('Number of fraudulent transactions')
plt.xlabel('Initial customers')
plt.grid(axis='x')

del ax, df2


# * Initial customers are scammed of **at most one** for a fraudulent transaction.

# In[24]:


df2 = df1['nameDest'].value_counts().head(10)
ax = df2.plot(kind='bar', color='lightsteelblue')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Top 10 recipients highly involved in fraudulent transactions')
plt.ylabel('Number of fraudulent transactions')
plt.xlabel('Recipients')
plt.grid(axis='x')

del ax, df2


# * Suspicious recipients only have a **maximum of 2** involved fraudulent transactions.

# In[25]:


df['oldbalanceDest_amt'] = pd.cut(df['oldbalanceDest'], 5, labels=['very low', 'low', 'moderate', 'high', 'very high'])

ax = sns.countplot(x='oldbalanceDest_amt', data=df, hue='isFraud', palette='PuBu')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Count plot of recipients pre-transaction balance amount')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.ylabel('Number of transactions')
plt.xlabel('Recipient pre-transaction balance amount')


# * Recipients with **very low pre-transaction balances** has the highest number of fraudulent transactions. 
# * This implies that recipients with very low pre-transaction balances may be more susceptible to fraudulent transactions.

# <h2>
#     <div style='background:steelblue'> 
#     <font color='white'>
#     <b>&nbsp; Multivariate data visualization</b>
#     </font>
#     </div>
# </h2>

# In[26]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr(method='spearman')

sns.heatmap(corr_matrix, cbar=True, annot=True, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)), fmt='.3f', cmap='PuBu')
plt.title('Correlation')
plt.show()


# * oldbalanceOrg and newbalanceOrig has **strong positive** relationship. 
# * oldbalanceDest and newbalanceDest has **strong positive** relationship.
# * oldbalanceOrg and amount has **weak positive** relationship.
# * newbalanceOrig and amount has **moderate positive** relationship.

# # <div style='background:darkblue'> <font color='white'> <b>&nbsp; Model Building</b> </font> </div>

# In[27]:


# Data preprocessing
df['type'] = df['type'].map({'PAYMENT':0, 'CASH_IN':1, 'DEBIT':2, 'CASH_OUT':3, 'TRANSFER':4})


# Due to the large dataset, Random Forest and Logistic Regression with balanced class weight are used to identify online payment fraud.

# In[28]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay


seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


X = df.copy()
X.drop(['nameOrig', 'newbalanceOrig', 'nameDest', 'newbalanceDest', 'quantity', 'oldbalanceOrg_amt', 'oldbalanceDest_amt'], axis=1, inplace=True)
y = X.pop('isFraud')


# Stratified train-test split
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train_idx, test_idx in skfold.split(X,y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


sc = StandardScaler()
scaled_train = sc.fit_transform(X_train)
scaled_test = sc.transform(X_test)
X_train = pd.DataFrame(scaled_train, index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaled_test, index=X_test.index, columns=X_test.columns)


X_train, y_train = RandomUnderSampler(sampling_strategy='majority').fit_resample(X_train, y_train)


# In[29]:


def model_comparison_evaluate(classifiers, X, y):
    print('K-Fold Cross-Validation:\n')
    for name, model in classifiers.items():
        print('{}:'.format(name))
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for score in scoring:
            scores = cross_val_score(model, X, y, scoring=score, cv=skfold, n_jobs=-1)
            print('Mean {} score: {:.3f} ({:.3f})'.format(score, scores.mean(), scores.std()))
            
        print('\n')


# In[30]:


classifiers = { 'Random Forest Classifier':RandomForestClassifier(class_weight='balanced', random_state=seed),
                'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=seed)
              }


# In[31]:


model_comparison_evaluate(classifiers, X_train, y_train)


# In[32]:


model = RandomForestClassifier(class_weight='balanced', random_state=seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_score = model.predict_proba(X_test)[:,1]
print('Random Forest Classifier:')
print(classification_report(y_pred, y_test, labels=[0,1], target_names=['Non-Fraud [0]', 'Fraud [1]']), '\n')



fig, ax = plt.subplots(1, 2, figsize=(20,5))
ax[0].set_title('Confusion Matrix of Random Forest Model:')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, values_format='', cmap='crest', ax=ax[0])
ax[0].grid(False)



fpr, tpr, thresholds = roc_curve(y_test, y_pred_score)
roc_auc = auc(fpr, tpr)                       
ax[1].set_title('ROC Curve - Random Forest Classifier')
ax[1].plot(fpr, tpr, label = 'AUC = %0.3f' % roc_auc, c='steelblue')
ax[1].plot([0,1],[0,1],'--', c='lightsteelblue')
ax[1].legend(loc='lower right')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_xlabel('False Positive Rate')


# * From the confusion matrix, 1,239,155 were correctly classified as non-fraudulent payments, and 31,726 people were misclassified as non-fraudulent payments.
# * According to the confusion matrix, 1,639 payments were incorrectly labelled as fraud while 4 payments were correctly identified as fraud.

# # <div style='background:darkblue'>  <font color='white'> <b>&nbsp; Conclusion</b> </font> </div>
# 
# * Random Forest obtains the highest score of all using K-fold cross-validation.
# * The best performing model is **Random Forest** for identifying fraudulent and non-fraudulent payments, as the AUC is 0.999, which is close to 1. This means it has a good separability measure, and the model has an 99.9% chance of being able to distinguish between positive and negative classes..

# In[33]:


get_ipython().system('jupyter nbconvert --to script online-payment-fraud-detection.ipynb')


# In[ ]:





# In[ ]:




