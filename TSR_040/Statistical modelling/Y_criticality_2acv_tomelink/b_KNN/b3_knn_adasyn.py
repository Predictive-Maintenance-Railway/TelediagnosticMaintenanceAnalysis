# a) Define the pipeline:
pipeline = Pipeline2([
    ('adasyn', ADASYN(random_state=123)),
    ('tomek', TomekLinks(sampling_strategy='majority')),
    ('knn', KNeighborsClassifier())
])

# b) Define the parameter grid
param_grid = {
    'adasyn__n_neighbors': range(3,5),
    'adasyn__sampling_strategy': np.arange(0.4, 1.1, 0.1),
    'knn__n_neighbors': np.arange(1,26,1),
    'knn__weights': ['uniform', 'distance'],
}
f1_scorer = make_scorer(f1_score)

# c) Tuning parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f1_scorer, refit=False, n_jobs=-1, error_score='raise')
result_tuning = grid_search.fit(x_train_sd, y_train)
# check value of hyperparameters:
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
#print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())
print(f"ADASYN n neighbors: {tuned_par[0]}")
print(f"Imbalance ratio: {tuned_par[1]}")
print(f"k: {tuned_par[2]}")
print(f"weight: {tuned_par[3]}")

# d) Apply ADASYN, tomeklinks and fit knn with tuned parameters
adasyn_tuned = ADASYN(sampling_strategy=tuned_par[1], n_neighbors=tuned_par[0], random_state=123)
x_train_sd_adasyn, y_train_adasyn = adasyn_tuned.fit_resample(x_train_sd, y_train)
tomek = TomekLinks(sampling_strategy='majority')
x_train_sd_adasyn_tomek, y_train_adasyn_tomek = tomek.fit_resample(x_train_sd_adasyn, y_train_adasyn)
print(y_train_adasyn_tomek.value_counts())
knn_tuned = KNeighborsClassifier(n_neighbors=tuned_par[2], weights=tuned_par[3])
knn_fit = knn_tuned.fit(x_train_sd_adasyn_tomek, y_train_adasyn_tomek)

# e) Predict on the test set
y_prob = knn_fit.predict_proba(x_test_sd)[:, 1]    # predicted prob for positive class
y_pred = knn_fit.predict(x_test_sd)

# f) Calculate performance metrics: f1 score, sensitivity, specificity, auc
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test,  y_prob)
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
ns_y_probs = [0 for _ in range(len(y_test))]    # generate a no skill prediction (majority class)
ns_auc = roc_auc_score(y_test, ns_y_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_y_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label=f'KNN with AUC={correct_auc:.4f}')
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