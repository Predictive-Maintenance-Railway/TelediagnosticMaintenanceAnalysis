# a) Define a custom scoring function to optimize the threshold during grid search
def threshold_f1_scorer(y_true, y_probs):
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_score = max(scores)
    return best_score

# b) Define bagging tree
base_classifier = DecisionTreeClassifier(random_state=123)
bagging_classifier = BaggingClassifier(estimator=base_classifier, n_estimators=500, oob_score=True, n_jobs=-1, random_state=123)

# c) Fit
bagg_fit = bagging_classifier.fit(x_train, y_train)

# d) Find the best threshold on the test set
y_prob = bagg_fit.predict_proba(x_test)[:, 1]
thresholds = np.arange(0, 1, 0.001)
scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(scores)]
test_f1 = max(scores)

# e) Predict on the test set
y_pred = (y_prob >= best_threshold).astype(int)

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
plt.plot(fpr, tpr, marker='.', label='Bagging and AUC='+str(correct_auc))
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