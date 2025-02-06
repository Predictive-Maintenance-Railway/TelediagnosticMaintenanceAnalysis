# a) Define the knn classifier
knn = KNeighborsClassifier(
    #p=2 by default, which means euclidean distance is used
)

# b) Define the parameter grid
param_grid = {
    'n_neighbors': np.arange(1,26,1),               # Number of neighbors to try
    'weights': ['uniform', 'distance'],             # Weights for neighbors
}
# and use F1 score as metric for tuning
f1_scorer = make_scorer(f1_score)

# c) Perform GridSearchCV to find the best parameter based on F1 score
grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, refit=False)
result_tuning = grid_search.fit(x_train_sd, y_train)
# check value of f1 score and C:
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
#print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())
print(f"k: {tuned_par[0]}")
print(f"weight: {tuned_par[1]}")

# d) fit knn with tuned parameter
knn_tuned = KNeighborsClassifier(n_neighbors=tuned_par[0], weights=tuned_par[1])
knn_fit = knn_tuned.fit(x_train_sd, y_train)

# e) Predict on the test set
y_prob = knn_fit.predict_proba(x_test_sd)[:, 1]    # predicted prob for positive class
y_pred = knn_fit.predict(x_test_sd)

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
plt.plot(fpr, tpr, marker='.', label='Knn and AUC='+str(correct_auc))
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