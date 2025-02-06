# a) Define a custom scoring function to optimize the threshold during grid search
def threshold_f1_scorer(y_true, y_probs):
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_score = max(scores)
    return best_score

# b) Define the Boosting classifier (AdaBoost)
base_classifier = DecisionTreeClassifier(random_state=123)
boost = AdaBoostClassifier(estimator=base_classifier, algorithm='SAMME', random_state=123)

# c) Define the parameter grid
param_grid = {
    'estimator__max_depth': range(1,5),
    'n_estimators': np.arange(10,100,10),
    'learning_rate': np.arange(0.01,0.1,0.01)
}

# c) Grid search with the custom scoring function
grid_search = GridSearchCV(boost, param_grid, cv=cv, scoring=make_scorer(threshold_f1_scorer, needs_proba=True), n_jobs=-1)
result_tuning = grid_search.fit(x_train, y_train)
# check value of hyperparameters:
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
#print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())
print(f"depth tree: {tuned_par[0]}")
print(f"learning rate: {tuned_par[1]}")
print(f"n. trees: {tuned_par[2]}")

# d) Fit boosting with tuned parameter
base_classifier_tuned = DecisionTreeClassifier(max_depth=tuned_par[0],random_state=123)
boost_tuned = AdaBoostClassifier(estimator=base_classifier_tuned, algorithm='SAMME', n_estimators=tuned_par[2], learning_rate=tuned_par[1], random_state=123)
boost_fit = boost_tuned.fit(x_train, y_train)

# e) Find the best threshold on the test set
y_prob = boost_fit.predict_proba(x_test)[:, 1]
thresholds = np.arange(0, 1, 0.001)
scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(scores)]
test_f1 = max(scores)

# f) Predict on the test set
y_pred = (y_prob >= best_threshold).astype(int)

# g) Calculate performance metrics: f1 score, sensitivity, specificity, auc
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test,  y_prob)
auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# h) Print out the results
print("\nModel fitted:")
print(f"Best threshold: {best_threshold}")
print(f"F1 Score: {f1:.6f}")
print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
correct_auc = (auc-0.5)/0.5
print(f"AUC: {correct_auc:.6f}")

# i) Plot roc curve:
ns_y_probs = [0 for _ in range(len(y_test))]    # generate a no skill prediction (majority class)
ns_auc = roc_auc_score(y_test, ns_y_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_y_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label=f'Boosting with AUC={correct_auc:.4f}')
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

# i) Variable importance plot
feature_importance = boost_fit.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(x_test.columns)[sorted_idx])
plt.xlabel('Mean decrease Gini')
plt.title('Variable importance Boosting with moving threshold')

# and Partial dependence plot
features = list(range(x_train.shape[1]))
fig, ax = plt.subplots(figsize=(20, 8))
display = PartialDependenceDisplay.from_estimator(boost_fit, x_train_adasyn, features, ax=ax, grid_resolution=20)
plt.show()