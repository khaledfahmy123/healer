import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score




training = pd.read_csv('Training.csv') # training (training , test)
# print(training)
# testing= pd.read_csv('Testing.csv') # final testing
# print(testing)
cols= training.columns

cols= cols[:-1]

x = training[cols]
y = training['prognosis']


#mapping strings to numbers
process = preprocessing.LabelEncoder()
process.fit(y)

y = process.transform(y)
# print(list(process.classes_))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)

scores = cross_val_score(clf, x_test, y_test, cv=3)

print ("Model Accuracy: ", scores)


# print(type(x[2:3]))
# print(clf.predict(x[2:3]))
# print(y[2:3])