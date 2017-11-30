import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
estimator = DecisionTreeClassifier(max_leaf_nodes=3)
#t0 = time()
estimator.fit(X_train,y_train)
#print("training time:",round(time()-t0,3),"s")
print estimator.score(X_test,y_test)

import graphviz
dot_data = tree.export_graphviz(estimator,out_file=None)
graph = graphviz.source(dot_data)
graph.render("iris")