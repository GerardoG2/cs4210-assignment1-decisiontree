#-------------------------------------------------------------------------
# AUTHOR: Gerardo Gutierrez
# FILENAME: decision_tree.py
# SPECIFICATION: Build Decision Tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: 45 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
age_dict        = {"Young":0, "Prepresbyopic":1, "Presbyopic":2}
spectacle_dict       = {"Myope":0, "Hypermetrope":1}
astigmatism_dict      = {"No":0, "Yes":1}
tear_dict      = {"Reduced":0, "Normal":1}
recommended_dict    = {"No":0, "Yes":1} 

for row in db:
   age = row[0]
   spectacle = row[1]
   astigmatism = row[2]
   tear = row[3]

   X.append([
      age_dict[age],
      spectacle_dict[spectacle],
      astigmatism_dict[astigmatism],
      tear_dict[tear],
   ])

#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
for row in db:
   recommended = row[4]
   
   Y.append([
      recommended_dict[recommended]
   ])

#fitting the decision tree to the data using entropy as your impurity measure

clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 42)
clf = clf.fit(X,Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()