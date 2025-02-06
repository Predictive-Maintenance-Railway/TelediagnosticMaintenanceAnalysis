# a) Define the decision tree
tree = DecisionTreeClassifier(random_state=123
                              # splitter=best by default, which means that best split at each node is chosen
                              # max_depth=None by default, so the tree is grown deep
                              # min_samples_split=2 by default, which is the minimum number of samples required to split an internal node
                              # min_samples_leaf=1 by default, which is the minimum number of samples required to be at a leaf node
                              # max_features=None by default, which means all features are considered when looking for the best split
                              # min_impurity_decrease=0 by default, which is: a node will be split if this split induces a decrease of the impurity greater than or equal to this value
)

# b) Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],        # measure of the quality of a split
    'ccp_alpha': np.arange(0, 1, 0.001),      # cost complexity parameter
}
# and use F1 score as metric for tuning
f1_scorer = make_scorer(f1_score)

# c) Perform GridSearchCV to find the best parameter based on F1 score
grid_search = GridSearchCV(tree, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, refit=False)
result_tuning = grid_search.fit(x_train, y_train)
# check value of f1 score and C:
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
#print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())
print(f"criterion: {tuned_par[1]}")
print(f"alpha: {tuned_par[0]}")
'''
# plot alpha vs n.of nodes and vs f1 score:
path = tree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_alphas_2 = np.arange(0, 1, 0.001)
clfs = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=123, ccp_alpha=ccp_alpha)
    tree.fit(x_train, y_train)
    clfs.append(tree)
node_counts = [tree.tree_.node_count for tree in clfs]
depth = [tree.tree_.max_depth for tree in clfs]
fig, ax = plt.subplots(1, 1)
ax.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post", markersize=3)
ax.set_xlabel("alpha")
ax.set_ylabel("number of nodes")
ax.set_title("Number of nodes vs alpha")

f1_scores = grid_search.cv_results_['mean_test_score']
ccp_alphas_tried = [params['ccp_alpha'] for params in grid_search.cv_results_['params']]
#print(len(ccp_alphas_tried))
#print(len(f1_scores))
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("f1 score")
ax.set_title("F1 score vs alpha")
ax.plot(ccp_alphas_tried, f1_scores, marker="o", label="train", drawstyle="steps-post", markersize=3)
plt.show()
'''

# d) fit tree with tuned parameter
tree_tuned = DecisionTreeClassifier(criterion=tuned_par[1], ccp_alpha=tuned_par[0], random_state=123)
tree_fit = tree_tuned.fit(x_train, y_train)

# e) Predict on the test set
y_prob = tree_fit.predict_proba(x_test)[:, 1]    # predicted prob for positive class
y_pred = tree_fit.predict(x_test)

# f) Calculate performance metrics: f1 score, sensitivity, specificity, auc
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test,  y_prob)
auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# g) Print out the results
print("\nModel fitted:")
print(f"F1 Score: {f1:.6f}")
print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
correct_auc = (auc-0.5)/0.5
print(f"AUC: {correct_auc:.6f}")

# h) Plot roc curve:
# generate a no skill prediction (majority class)
ns_y_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_y_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_y_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Tree and AUC='+str(correct_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

thresholds_2 = np.arange(0.0, 1.01, 0.01)
fprs = []
tprs = []
for threshold_2 in thresholds_2:
    y_pred = (y_prob >= threshold_2).astype(int)  # Convert probabilities to binary predictions
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_2 = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_2 = tp / (tp + fn) if (tp + fn) > 0 else 0
    fprs.append(fpr_2)
    tprs.append(tpr_2)
plt.figure(figsize=(6.5, 6.5))
plt.plot(fprs, tprs, marker='.', label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='No skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Manually Specified Thresholds')
plt.legend()
plt.show()
