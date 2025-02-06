# a) Define the logistic regression model with L1 penalty
logistic = LogisticRegression(penalty='l1',             # =Lasso
                              solver='liblinear',
                              max_iter=1000,            # Maximum number of iterations taken for the solvers to converge --> increased because otherwise the algorithm doesn't find a solution
                              random_state=123)

# b) Define the parameter grid for tuning the regularization parameter C
param_grid = {'C': np.arange(0.05, 2, 0.05)}
# and use F1 score as metric for tuning
f1_scorer = make_scorer(f1_score)

# c) Perform GridSearchCV to find the best parameter based on F1 score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
grid_search = GridSearchCV(logistic, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, refit=False)
result_tuning = grid_search.fit(x_train, y_train)
# check value of f1 score and C:
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
tuned_par = list(result_tuning.best_params_.values())[0]
print(f"C tuned: {tuned_par}")

# d) fit logistic with tuned parameter
logistic_tuned = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, C=tuned_par, random_state=123)
log_fit = logistic_tuned.fit(x_train, y_train)

# e) Predict on the test set
y_prob = log_fit.predict_proba(x_test)[:, 1]    # predicted prob for positive class
y_pred = log_fit.predict(x_test)

# f) Calculate performance metrics: f1 score, sensitivity, specificity, auc
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
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

coefficients = log_fit.coef_[0]
non_zero_coefficients = coefficients != 0
used_features = np.array(non_zero_coefficients)

print("\nFeatures used in the model:")
for i, is_used in enumerate(non_zero_coefficients):
    if is_used:
        print(f"Feature {i+1} (Coefficient: {coefficients[i]:.4f})")
    else:
        print(f"Feature {i+1} (Coefficient: {coefficients[i]:.4f})")

# h) Plot roc curve:
# generate a no skill prediction (majority class)
ns_y_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_y_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_y_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic and AUC='+str(correct_auc))
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