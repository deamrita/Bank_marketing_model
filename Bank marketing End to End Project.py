#!/usr/bin/env python
# coding: utf-8

# ## Importing required libraries

# In[132]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# ## Importing data

# In[134]:


data = pd.read_csv("C:/Users/Amrita/OneDrive/Desktop/bank_marketing.csv",delimiter = ";")
print(data.head())
print("length of the data is:", len(data))


# ## EDA

# In[135]:


data.info()


# In[136]:


data.describe()


# In[137]:


data.rename(columns = {"y":"final outcome"},inplace = True)


# In[138]:


data.columns


# ## Checking null values

# In[139]:


data.isna().sum()


# In[140]:


#There is no null values


# ## Visualizing correlation between features

# In[142]:


data.corr()


# In[143]:


plt.figure(figsize = (5,5))
sns.heatmap(data.corr(),annot = True, fmt = ".0%")
plt.show()


# In[145]:


# There is no highly correlated features


# In[146]:


sns.displot(data["final outcome"], label = True)


# ## Outlier detection
# 

# In[148]:


# #Detecting outliers using z-score (no applicable here because the distribution of the variable is not normal)
# for column in data.columns:
#         if data[column].dtype == "int64":
#             z_scores = np.abs(stats.zscore(data[column]))
#             threshold = 2  # Adjust the threshold value based on your dataset
#             outliers = np.where(z_scores > threshold)
#             df = data.drop(outliers[0])
# len(df)


# In[149]:


#before removing outliers
for column in data.columns:
        if data[column].dtype == "int64":
            plt.figure(figsize=(16,8))
            plt.subplot(2,2,1)
            sns.distplot(data[column])
            plt.subplot(2,2,2)
            sns.boxplot(data[column])


# In[150]:


data.columns[:13]


# In[151]:


#removing outliers (using IQR)

for column in data.columns[:13]:
        if data[column].dtype == "int64":
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            data = data.drop(outliers.index)


# In[152]:


#after removing outliers
for column in data.columns[:13]:
        if data[column].dtype == "int64":
            plt.figure(figsize=(16,8))
            plt.subplot(2,2,1)
            sns.distplot(data[column])
            plt.subplot(2,2,2)
            sns.boxplot(data[column])


# In[153]:


print("Length of the data after removing outliers: ",len(data))


# ## Printing all of the object data types and their unique values

# In[154]:


for column in data.columns:
    if data[column].dtype == object:
        print(str(column) + ' : ' + str(data[column].unique()))
        print(data[column].value_counts())
        print("_________________________________________________________________")


# In[155]:


# 0/1 = final outcome
# one hot encoding = job, marital, contact, poutcome
# label encoder = month,education,default, housing, loan


# In[156]:


# Mapping the target variable to 0 and 1
data["final outcome"] = data["final outcome"].replace(['yes','no'],['1','0'])
data.head()


# ## Encoding the non-numeric columns

# In[157]:


## Transform non-numeric columns into numerical columns
# One-Hot Encoding and Label Encoding are both techniques used for converting categorical variables into a numerical representation. However, they differ in their approach and usage:

# 1. Label Encoding:
#    - Label Encoding assigns a unique numerical label to each unique category in a categorical variable.
#    - It replaces categorical values with integer labels, typically ranging from 0 to (number of categories - 1).
#    - Label Encoding is suitable for ordinal variables where the categories have an inherent order or rank.
#    - The encoded labels may introduce an arbitrary ordinal relationship between the categories that may mislead the model.
#    - Example: Label Encoding the categories "red," "green," and "blue" as 0, 1, and 2, respectively.

# 2. One-Hot Encoding:
#    - One-Hot Encoding creates binary columns for each category in a categorical variable.
#    - It represents each category as a binary vector (0 or 1) in a separate column.
#    - Each column represents one category, and only one of the columns is 1 for a particular instance.
#    - One-Hot Encoding is suitable for nominal variables where the categories do not have any inherent order.
#    - It avoids introducing arbitrary ordinal relationships between categories and prevents misinterpretation by the model.
#    - Example: One-Hot Encoding the categories "red," "green," and "blue" as three separate columns: [1, 0, 0], [0, 1, 0], and [0, 0, 1], respectively.

# When to use which one:
# - Label Encoding is typically used for ordinal variables where the order or rank of the categories is meaningful. For example, educational levels (high school, bachelor's degree, master's degree) or ratings (low, medium, high).
# - One-Hot Encoding is suitable for nominal variables where the categories have no inherent order or rank. It is commonly used when the presence or absence of a category is relevant to the problem. For example, colors (red, green, blue) or countries (USA, Canada, France).

# It's important to note that the choice between Label Encoding and One-Hot Encoding depends on the specific dataset, the nature of the categorical variable, and the machine learning algorithm being used. 
# Some algorithms can handle categorical variables directly, while others may require categorical variables to be converted into a numerical representation. 
#It's recommended to consider the characteristics of your data and the requirements of your machine learning model when deciding which encoding technique to use.


# In[158]:


#Transform non-numeric columns into numerical columns
#One hot encoding

#encoded_data = pd.concat([df4,pd.get_dummies(df4[['job','marital','contact','poutcome']],prefix=[['job','marital','contact','poutcome']])],axis=1).drop([['job','marital','contact','poutcome']],axis=1)

one_hot_encoded_data = pd.get_dummies(data, columns = ['job','marital','contact','poutcome'])


# In[159]:


one_hot_encoded_data.head()


# In[160]:


one_hot_encoded_data.columns


# In[161]:


data_encoded = pd.merge(data, one_hot_encoded_data, how = "inner")
print(data_encoded.columns)
data_encoded.drop(['job','marital','contact','poutcome'], axis=1, inplace=True)
print(data_encoded.columns)
print(len(data_encoded.columns))


# In[162]:


data_encoded.head()


# In[163]:


#Label encoding
#Transform non-numeric columns into numerical columns
from sklearn.preprocessing import LabelEncoder

for column in data_encoded.columns:
        if data_encoded[column].dtype == np.number:
            continue
        data_encoded[column] = LabelEncoder().fit_transform(data_encoded[column])


# In[164]:


data_encoded.head()


# In[165]:


len(data_encoded)


# In[166]:


print(len(data_encoded.columns))


# In[167]:


#Rearranging the columns
data_encoded = data_encoded[['final outcome','age', 'education', 'default', 'balance', 'housing', 'loan', 'day',
       'month', 'duration', 'campaign', 'pdays', 'previous',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'contact_cellular', 'contact_telephone', 'contact_unknown',
       'poutcome_failure', 'poutcome_other', 'poutcome_success',
       'poutcome_unknown']]


# ## Spliting the data into independent 'X' and dependent 'Y' variables

# In[169]:


X = data_encoded.iloc[:, 1:data_encoded.shape[1]]
Y = data_encoded.iloc[:, 0]


# ## Nomalization and Standardization

# In[170]:


# Normalization/ Min- Max scaling 
# Rescales values to a range between 0 and 1
# Useful when the distribution of the data is unknown or not Gaussian
# Sensitive to outliers
# Retains the shape of the original distribution
# Equation: (x – min)/(max – min)
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
X_norm = MinMaxScaler().fit_transform(X.values)
normed_features_df = pd.DataFrame(X_norm, index=X.index, columns=X.columns)


# In[171]:


#Standardization (Not applicable here)
# Centers data around the mean and scales to a standard deviation of 1
# Useful when the distribution of the data is Gaussian or unknown
# Less sensitive to outliers
# Changes the shape of the original distribution
# Equation: (x – mean)/standard deviation
# scaled_features = StandardScaler().fit_transform(X.values)
# scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)


# In[172]:


normed_features_df


# ## Train test split

# In[173]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(normed_features_df, Y, test_size=0.2, random_state=42)


# In[174]:


X_train.shape


# In[175]:


X_test.shape


# ## Implementing lightgbm

# In[176]:


import lightgbm as ltb

model1 = ltb.LGBMClassifier()
model1.fit(X_train, Y_train)
model1.score(X_test,Y_test)


# In[177]:


y_pred_1 = model1.predict(X_test)
y_pred_1


# In[178]:


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(Y_test, y_pred_1)


# In[179]:


# sensitivity, specificity and accuracy 

## Sensitivity/Recall : It is a measure of how well a machine learning model can detect positive instances.
## Specificity : It is the proportion of true negatives that are correctly predicted by the model.
## Accuracy : Determine which model is best at identifying relationships and patterns b/w variables based on input data.

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

cm = confusion_matrix(Y_test, y_pred_1)
total=sum(sum(cm))
Accuracy = (cm[0,0]+cm[1,1])/total
Specificity = cm[0,0]/(cm[0,0]+cm[0,1])   # 
Sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])

print('Accuracy of lightgbm', Accuracy)
print('Specificity of lightgbm', Specificity)
print('Sensitivity or Recall of lightgbm', Sensitivity)


# In[180]:


from sklearn.metrics import *

#calculate F1 score
f1_score = f1_score(Y_test, y_pred_1)

print('F1 score of lightgbm', f1_score)


# ## Implementing Gradient Boosting Classifier

# In[181]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=500,learning_rate=0.05,random_state=100,max_features=5)
gbc.fit(X_train,Y_train)
gbc.score(X_train,Y_train)


# In[182]:


y_pred_2 = gbc.predict(X_test)
y_pred_2


# In[183]:


cm2 = confusion_matrix(Y_test, y_pred_2)
total=sum(sum(cm2))
Accuracy = (cm2[0,0]+cm2[1,1])/total
Specificity = cm2[0,0]/(cm2[0,0]+cm2[0,1])   # 
Sensitivity = cm2[1,1]/(cm2[1,0]+cm2[1,1])

print('Accuracy of gbc', Accuracy)
print('Specificity of gbc', Specificity)
print('Sensitivity or Recall of gbc', Sensitivity)


# In[184]:


#calculate F1 score
f1_score2 = f1_score(Y_test, y_pred_2)

print('F1 score of gbc', f1_score2)


# ## Implementing Random Forest Classifier

# In[185]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
forest.score(X_train,Y_train)


# In[186]:


y_pred_3 = forest.predict(X_test)
y_pred_3


# In[187]:


cm3 = confusion_matrix(Y_test, y_pred_3)
total=sum(sum(cm3))
Accuracy = (cm3[0,0]+cm3[1,1])/total
Specificity = cm3[0,0]/(cm3[0,0]+cm3[0,1])   # 
Sensitivity = cm3[1,1]/(cm3[1,0]+cm3[1,1])

print('Accuracy of random forest', Accuracy)
print('Specificity of random forest', Specificity)
print('Sensitivity or Recall of random forest', Sensitivity)


# In[188]:


#calculate F1 score
f1_score3 = f1_score(Y_test, y_pred_3)

print('F1 score of random forest', f1_score3)


# ## Hyper parameter tuning using GridsearchCV on Random Forest

# In[59]:


# Define the parameter grid to search over
### We try every combination of a present list of values of the hyper-parameters and choose the best combination based on the cross validation score.
### - It takes a lot of time to fit (because it will try all the combinations)
### + gives us the best hyper-parameters.
### exemple ;
### { 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf',’linear’,'sigmoid'] }

### in this case we will try 5 * 5 * 3=75 combinations -->

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, Y_train)

# Print the best parameters and the corresponding score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[ ]:


# Evaluate the model on the test set using the best parameters
best_rf_classifier = grid_search.best_estimator_
test_accuracy = best_rf_classifier.score(X_test, Y_test)
print("Test accuracy: ", test_accuracy)


# ## Hyper parameter tuning using RandomsearchCV on Random Forest

# In[189]:


from sklearn.model_selection import RandomizedSearchCV
### Tries random combinations of a range of values (we have to define the number of iterations). It is good at testing a wide range of values and normally it reaches a very good combination very fast, but the problem that it doesn’t guarantee to give the best parameter combination because not all parameter values are tried out (recommended for big datasets or high number of parameters to tune.

### It doesn't guarantee that we have the best parameters
### faster because not all parameter values are tried out

param_dist = {"max_depth":  [None] + list(np.arange(5, 30, 5)), 
    "max_features": ['auto', 'sqrt', 'log2', None], 
    "min_samples_split": np.arange(2, 20, 2), 
    "bootstrap": [True, False], 
    "criterion": ["gini", "entropy"],
    'min_samples_leaf': np.arange(1, 10, 1)}

random_search = RandomizedSearchCV(forest, param_distributions=param_dist, n_iter=20, cv=5,scoring='accuracy') 

random_search.fit(X_train, Y_train)
# Print the best parameters and the corresponding score
print("Best parameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)


# In[190]:


# Evaluate the model on the test set using the best parameters
best_rf_classifier2 = random_search.best_estimator_
test_accuracy = best_rf_classifier2.score(X_test, Y_test)
print("Test accuracy: ", test_accuracy)


# ## Feature importance

# In[191]:


# Return the feature importances (the higher, the more important the feature).
importances = pd.DataFrame({'feature':data_encoded.iloc[:, 1:data_encoded.shape[1]].columns,'importance':np.round(best_rf_classifier2.feature_importances_,3)}) #Note: The target column is at position 0
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.reset_index(inplace = True)
print(importances)


# In[192]:


fig, ax = plt.subplots(1, figsize = (20, 8))
ax.grid()
fig.autofmt_xdate() #to prevent overlapping in x axis labels

plt.bar(importances.feature, importances.importance, color ='maroon',
        width = 0.4)

plt.show()


# ## Pickling the model file for deployment

# In[194]:


import pickle

pickle.dump(best_rf_classifier2,open('bankmodel_v2.pkl','wb'))


# In[195]:


pickled_model = pickle.load(open('bankmodel_v2.pkl','rb'))


# ## Predicting the unseen data

# In[196]:


#importing the data

unseen_data = pd.read_csv("C:/Users/Amrita/OneDrive/Desktop/bank_test_unseen.csv",delimiter = ";")


# In[197]:


unseen_data.head()


# In[198]:


X_test.head(2)


# ## Encoding and scaling the unseen the data

# In[199]:


one_hot_encoded_test = pd.get_dummies(unseen_data, columns = ['job','marital','contact','poutcome'])

data_encoded_unseen = pd.merge(unseen_data, one_hot_encoded_test, how = "inner")
print(data_encoded_unseen.columns)
data_encoded_unseen.drop(['job','marital','contact','poutcome'], axis=1, inplace=True)
print(data_encoded_unseen.columns)
print(len(data_encoded_unseen.columns))


#Transform non-numeric columns into numerical columns

for column in data_encoded_unseen.columns:
        if data_encoded_unseen[column].dtype == np.number:
            continue
        data_encoded_unseen[column] = LabelEncoder().fit_transform(data_encoded_unseen[column])


# In[200]:


data_encoded_unseen = data_encoded_unseen[['y','age', 'education', 'default', 'balance', 'housing', 'loan', 'day',
       'month', 'duration', 'campaign', 'pdays', 'previous',
       'job_entrepreneur', 'job_management', 'job_technician',
       'marital_married', 'marital_single', 'contact_unknown',
       'poutcome_unknown']]


# In[201]:


X = data_encoded_unseen.iloc[:, 1:data_encoded_unseen.shape[1]]
Y = data_encoded_unseen.iloc[:, 0]


# In[202]:


X.isnull().sum()


# In[203]:


X_norm_unseen = MinMaxScaler().fit_transform(X.values)
normed_features_df = pd.DataFrame(X_norm_unseen, index=X.index, columns=X.columns)


# In[204]:


normed_features_df.columns


# In[205]:


list_n = list(set(X_test.columns) - set(normed_features_df.columns))


# ## Filling the not present columns with dummy value 0

# In[206]:


for i in list_n:
    normed_features_df[i] = 0


# In[207]:


print(len(normed_features_df.columns))
print(len(X_test.columns))


# In[208]:


normed_features_df = normed_features_df[X_test.columns]


# In[209]:


normed_features_df.head()


# In[210]:


prediction = best_rf_classifier2.predict(normed_features_df)


# In[211]:


prediction


# In[212]:


pickled_model.predict(normed_features_df)


# In[ ]:




