
# coding: utf-8

# In[57]:

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
from io import StringIO


# In[58]:

iris = load_iris()


# In[59]:

test_idx = [0,50,100]


# In[60]:

#training 
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx , axis=0)


# In[61]:

#testing 
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


# In[62]:

clf = tree.DecisionTreeClassifier()


# In[63]:

clf.fit(train_data , train_target)


# In[64]:

test_target


# In[65]:

clf.predict(test_data)


# In[66]:

dot_data = StringIO()
tree.export_graphviz(clf,out_file=dot_data, feature_names = iris.feature_names, class_names= iris.target_names, filled=True , rounded=True, impurity=False) 


# In[67]:

dot_data.getvalue()
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("iris.pdf")


# In[ ]:



