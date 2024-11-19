import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
import json
import seaborn as sns
import matplotlib.pyplot as plt

# importing alll the necessary packages to use the various classification algorithms
from sklearn.metrics import confusion_matrix
import joblib
import os
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.neighbors import KNeighborsClassifier # KNN classifier
from sklearn import svm # for suport vector machine algorithm
from sklearn import metrics # for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier # for using DTA
import dvc.api

iris = pd.read_csv("data_raw.csv")

params = dvc.api.params_show()
epochs = params['epochs']
print(f'The value of the environment variable "epoch" is: {epochs}')

train, test = train_test_split(iris, test_size=0.3, random_state=1) # our main data split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
print(train.shape)
print(test.shape)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking the training data features
train_y = train.Species # output of the training data

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data feature
test_y = test.Species # output value of the test data

petal = iris[['PetalLengthCm','PetalWidthCm','Species']]
sepal = iris[['SepalLengthCm','SepalWidthCm','Species']]

train_p,test_p = train_test_split(petal, test_size=0.3, random_state=0) #petals
train_x_p = train_p[['PetalWidthCm','PetalLengthCm']]
train_y_p = train_p.Species

test_x_p = test_p[['PetalWidthCm','PetalLengthCm']]
test_y_p = test_p.Species

train_s,test_s = train_test_split(sepal, test_size=0.3, random_state=0) #sepals
train_x_s = train_s[['SepalWidthCm','SepalLengthCm']]
train_y_s = train_s.Species

test_x_s = test_s[['SepalWidthCm','SepalLengthCm']]
test_y_s = test_s.Species


model = svm.SVC() # select the svm algorithm

# we train the algorithm with training data and training output
model.fit(train_X, train_y)

# we pass the testing data to the stored algorithm to predict the outcome
prediction = model.predict(test_X)
accuracy_all = metrics.accuracy_score(prediction, test_y)
print('The accuracy of the SVM is: ', accuracy_all)# we check the accuracy of the algorithm

#save the trained model
joblib.dump(model, 'model_all.pkl')

model=svm.SVC()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
accuracy_petal = metrics.accuracy_score(prediction, test_y_p)
print('The accuracy of the SVM using Petals is:',accuracy_petal)

#save the trained model
joblib.dump(model, 'model_petal.pkl')

model=svm.SVC()
model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
accuracy_sepal = metrics.accuracy_score(prediction, test_y_s)
print('The accuracy of the SVM using Sepals is:',accuracy_sepal)

# Save the model to a file
joblib.dump(model, 'model_sepal.pkl')

# model = LogisticRegression()
# model.fit(train_X, train_y)
# prediction = model.predict(test_X)
# accuracy_petal = metrics.accuracy_score(prediction, test_y)
# print('The accuracy of the LR using Petals is:',accuracy_petal)

# model = LogisticRegression()
# model.fit(train_x_p,train_y_p) 
# prediction=model.predict(test_x_p) 
# accuracy_petal = metrics.accuracy_score(prediction, test_y_p)
# print('The accuracy of the Logistic Regression using Petals is:',accuracy_petal)

# model = LogisticRegression()
# model.fit(train_x_s,train_y_s) 
# prediction=model.predict(test_x_s) 
# accuracy_sepal = metrics.accuracy_score(prediction, test_y_s)
# print('The accuracy of the Logistic Regression using Sepals is:',accuracy_sepal)


# model = DecisionTreeClassifier()
# model.fit(train_X, train_y)
# prediction = model.predict(test_X)
# print('The accuracy of Decision Tree is: ', metrics.accuracy_score(prediction, test_y))

# model = KNeighborsClassifier(n_neighbors=3) # this examines 3 neighbors for putting the data into class
# model.fit(train_X, train_y)
# prediction = model.predict(test_X)

# print('The accuracy of KNN is: ', metrics.accuracy_score(prediction, test_y))



# Now print to file
with open("metrics.json", 'w') as outfile:
        json.dump({ "complete": accuracy_all, "petal": accuracy_petal, "sepal": accuracy_sepal}, outfile)

# model=DecisionTreeClassifier()
# model.fit(train_x_p,train_y_p) 
# prediction=model.predict(test_x_p) 
# print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))

# model.fit(train_x_s,train_y_s) 
# prediction=model.predict(test_x_s) 
# print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))

# model=KNeighborsClassifier(n_neighbors=3) 
# model.fit(train_x_p,train_y_p) 
# prediction=model.predict(test_x_p) 
# print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))

# model.fit(train_x_s,train_y_s) 
# prediction=model.predict(test_x_s) 
# print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))

# Generate a bar plot for the accuracies
accuracies = {
    'Complete': accuracy_all,
    'Petal': accuracy_petal,
    'Sepal': accuracy_sepal
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
#plt.xlabel('Model Type')
plt.ylabel('Accuracy')
plt.title('Accuracy of LR Model on Different Features')
plt.ylim(0, 1)
plt.savefig('by_feature.png')