
clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 30, max_features ='log2', splitter = 'random', max_depth = 7, min_samples_leaf = 10, min_samples_split = 70 )
score = clf.fit(X_train, y_train)
predict_test_y = clf.predict(X_test0)
imp = clf.feature_importances_
 
res = clf.predict(Xtrans_test)
