from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import graphviz
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
estimator = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=8)
#t0 = time()
estimator.fit(X_train, y_train)
#print("training time:", round(time()-t0, 3), "s")
print(estimator.feature_importances_)

dot_data = tree.export_graphviz(estimator, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
#dot_data = tree.export_graphviz(estimator, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris",view=True)
print estimator.score(X_test, y_test)

estimator.fit(X_train,y_train)
y_pred=estimator.predict(X_test)

print ("acuraccy from DT after stan")
score =accuracy_score(y_test,y_pred)

print score
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)





                         
                         

