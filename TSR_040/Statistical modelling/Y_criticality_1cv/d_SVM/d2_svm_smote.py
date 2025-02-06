# a) Define the pipeline
pipeline = Pipeline2([
    ('smote', SMOTE(random_state=123)),
    ('svm', SVC(random_state=123))
])

# b) Define the parameter grid for different kernels
param_grid = [
    {
        'smote__k_neighbors': range(3,5),
        'smote__sampling_strategy': np.arange(0.4, 1.1, 0.1),
        'svm__kernel': ['linear'],
        'svm__C': [0.001,0.01,0.1,0.2,0.5,1,2,5,10]
    },
    {
        'smote__k_neighbors': range(3,5),
        'smote__sampling_strategy': np.arange(0.4, 1.1, 0.1),
        'svm__kernel': ['poly'],
        'svm__C': [0.001,0.01,0.1,0.2,0.5,1,2,5,10],
        'svm__degree': [2, 4],
        'svm__coef0': [0.0, 0.1, 0.5, 1.0]
    },
    {
        'smote__k_neighbors': range(3,5),
        'smote__sampling_strategy': np.arange(0.4, 1.1, 0.1),
        'svm__kernel': ['rbf'],
        'svm__C': [0.001,0.01,0.1,0.2,0.5,1,2,5,10],
        'svm__gamma': ['scale', 'auto', 0.5,1,2,3,4,5, 10]
    }
]
f1_scorer = make_scorer(f1_score)

# c) Tuning parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, refit=False)
result_tuning = grid_search.fit(x_train_sd, y_train)
# check value of hyperparameters:
print("Tuning parameter:")
print(f"F1 score max: {result_tuning.best_score_}")
print(result_tuning.best_params_)
tuned_par = list(result_tuning.best_params_.values())

# d) Apply SMOTE and fit svm with tuned parameter
smote_tuned = SMOTE(sampling_strategy=tuned_par[1], k_neighbors=tuned_par[0], random_state=123)
x_train_sd_smote, y_train_smote = smote_tuned.fit_resample(x_train_sd, y_train)
print(y_train_smote.value_counts())
svm_tuned = SVC(C=tuned_par[2], kernel=tuned_par[3], probability=True, random_state=123)
svm_fit = svm_tuned.fit(x_train_sd_smote, y_train_smote)

# e) Predict on the test set
y_prob = svm_fit.predict_proba(x_test_sd)[:, 1]    # predicted prob for positive class
y_pred = svm_fit.predict(x_test_sd)

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




'''
def optimize_svc_with_smote(kernel_type, x_train, y_train):
    # Define the pipeline
    pipeline_2 = Pipeline([
        ('smote', SMOTE(random_state=123)),
        ('svm', SVC())
    ])

    # Define the parameter grid based on the kernel type
    if kernel_type == 'linear':
        param_grid = {
            'smote__k_neighbors': range(3, 5),
            'smote__sampling_strategy': np.arange(0.4, 1.1, 0.1),
            'svm__kernel': ['linear'],
            'svm__C': [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        }
    elif kernel_type == 'poly':
        param_grid = {
            'smote__k_neighbors': range(3, 5),
            'smote__sampling_strategy': np.arange(0.4, 1.1, 0.1),
            'svm__kernel': ['poly'],
            'svm__C': [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
            'svm__degree': [2, 4],                                  # con 3 ci sta milioni di anni....non so perchè
            'svm__coef0': [0.0, 0.1, 0.5, 1.0]
        }
    elif kernel_type == 'rbf':
        param_grid = {
            'smote__k_neighbors': range(3, 5),
            'smote__sampling_strategy': np.arange(0.4, 1.1, 0.1),
            'svm__kernel': ['rbf'],
            'svm__C': [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
            'svm__gamma': ['scale', 'auto', 0.5, 1, 2, 3, 4, 5, 10]
        }
    else:
        raise ValueError(f"Kernel type '{kernel_type}' is not supported. Choose from 'linear', 'poly', or 'rbf'.")

    # Use F1 score as the evaluation metric
    f1_scorer = make_scorer(f1_score)

    # Define the StratifiedKFold cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # Set up GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, verbose=0, refit=False)

    # Fit the model on the training data
    result_tuning = grid_search.fit(x_train, y_train)

    # Print the best parameters and best F1 score
    print(f"Best parameters for {kernel_type} kernel: {result_tuning.best_params_}")
    print(f"Best cross-validation F1 score for {kernel_type} kernel: {result_tuning.best_score_}")

    # Extract and return the best parameters
    return result_tuning.best_params_, result_tuning.best_score_

# Optimize for the linear kernel
best_linear_params, best_linear_f1 = optimize_svc_with_smote('linear', x_train_sd, y_train)

# Optimize for the polynomial kernel
best_poly_params, best_poly_f1 = optimize_svc_with_smote('poly', x_train_sd, y_train)

# Optimize for the RBF kernel
best_rbf_params, best_rbf_f1 = optimize_svc_with_smote('rbf', x_train_sd, y_train)
'''