# a) Create the base classifier
base_classifier = DecisionTreeClassifier(random_state=123)

# b) Create the Bagging classifier
bagging_classifier = BaggingClassifier(estimator=base_classifier,
                                       n_estimators=500,            #n_estimators, this is the number of base classifiers that our model is going to aggregate together --> we don't tune it because the n. of tree is not a critical parameter with bagging and what is usually done is to set a value sufficiently large: 500 (default in R)
                                       # max_features=1.0 by default , which means all features are used
                                       # max_samples=1.0 by default, which means using all samples drawn from the training set to train each base estimator
                                       # bootstrap=True by default, which means that samples are drawn with replacement
                                       # bootstrap_features=False by default, meaning all features are used for each base estimator
                                       oob_score=True,                      # use out-of-bag samples to estimate the generalization error
                                       n_jobs=-1,
                                       random_state=123)

# c) Fit
bagg_fit = bagging_classifier.fit(x_train, y_train)

# d) Predict on the test set
y_prob = bagg_fit.predict_proba(x_test)[:, 1]    # predicted prob for positive class
y_pred = bagg_fit.predict(x_test)

# e) Calculate performance metrics: f1 score, sensitivity, specificity, auc
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test,  y_prob)
auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# f) Print out the results
print("\nModel fitted:")
print(f"F1 Score: {f1:.6f}")
print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
correct_auc = (auc-0.5)/0.5
print(f"AUC: {correct_auc:.6f}")

# g) Plot roc curve:
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