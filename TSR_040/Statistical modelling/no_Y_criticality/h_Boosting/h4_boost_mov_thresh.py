# a) Define a custom scoring function to optimize the threshold during grid search
def threshold_f1_scorer(y_true, y_probs):
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_score = max(scores)
    return best_score

# b) Create the Boosting classifier (AdaBoost)
base_classifier = DecisionTreeClassifier(random_state=123)
boost = AdaBoostClassifier(estimator=base_classifier, algorithm='SAMME', random_state=123)

# c) Define the parameter grid
param_grid = {
    'estimator__max_depth': range(1,5),
    'n_estimators': np.arange(10,100,10),
    'learning_rate': np.arange(0.01,0.1,0.01)
}
# and use F1 score as metric for tuning
f1_scorer = make_scorer(f1_score)

# c) Grid search with the custom scoring function
grid_search = GridSearchCV(boost, param_grid, cv=5, scoring=make_scorer(threshold_f1_scorer, needs_proba=True), n_jobs=-1)
result_tuning = grid_search.fit(x_train, y_train)
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
#print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())
print(f"depth tree: {tuned_par[0]}")
print(f"learning rate: {tuned_par[1]}")
print(f"n. trees: {tuned_par[2]}")

# d) fit boosting with tuned parameter
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
print(f"F1 Score: {f1:.6f}")
print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
correct_auc = (auc-0.5)/0.5
print(f"AUC: {correct_auc:.6f}")

# i) Plot roc curve:
# generate a no skill prediction (majority class)
ns_y_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_y_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_y_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Boosting and AUC='+str(correct_auc))
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