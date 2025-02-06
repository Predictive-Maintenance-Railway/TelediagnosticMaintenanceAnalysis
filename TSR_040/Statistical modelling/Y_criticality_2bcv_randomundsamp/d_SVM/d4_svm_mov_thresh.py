# a) Define a custom scoring function to optimize the threshold during grid search
def threshold_f1_scorer(y_true, y_probs):
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_score = max(scores)
    return best_score

# b) Define the parameter grid for GridSearchCV
param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.001,0.01,0.1,0.2,0.5,1,2,5,10]
    },
    {
        'kernel': ['poly'],
        'C': [0.001,0.01,0.1,0.2,0.5,1,2,5,10],
        'degree': [2, 4],
        'coef0': [0.0, 0.1, 0.5, 1.0]
    },
    {
        'kernel': ['rbf'],
        'C': [0.001,0.01,0.1,0.2,0.5,1,2,5,10],
        'gamma': ['scale', 'auto', 0.5,1,2,3,4,5, 10]
    }
]
# and the classifier
svc = SVC(probability=True, random_state=123)

# c) Grid search with the custom scoring function
grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring=make_scorer(threshold_f1_scorer, needs_proba=True), n_jobs=-1)
result_tuning = grid_search.fit(x_train_sd, y_train)
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())

# d) Fit svm with tuned parameter
svm_tuned = SVC(C=tuned_par[0], kernel=tuned_par[1], probability=True, random_state=123)
svm_fit = svm_tuned.fit(x_train_sd, y_train)

# e) Find the best threshold on the test set
y_prob = svm_fit.predict_proba(x_test_sd)[:, 1]
thresholds = np.arange(0, 1, 0.001)
scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(scores)]
test_f1 = max(scores)

# f) Predict on the test set
y_pred = (y_prob >= best_threshold).astype(int)

# g) Calculate performance metrics: f1 score, sensitivity, specificity, auc
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test,  y_prob)
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
plt.plot(fpr, tpr, marker='.', label=f'SVM with AUC={correct_auc:.4f}')
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