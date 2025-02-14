# TelediagnosticMaintenanceAnalysis

### Overview

This repository contains the code used to perform the statistical analysis conducted as part of my thesis ["Predictive maintenance for railways: a case study"](https://iris.cnr.it/handle/20.500.14243/514952). Below the abstract:

Equipment failures, unplanned downtimes and the availability of spare parts significantly impact businesses that rely on assets, often resulting in production shutdowns and increased repair costs. Effective maintenance activities are essential for preventing such issues, and well-designed maintenance strategies can lead to reduced costs while enhanc- ing the efficiency and reliability of services. The railway sector, in particular, necessitates a substantial amount of maintenance to ensure smooth operations. This study presents the specific methodology developed to leverage various data provided by Trenord company to assess the feasibility of implementing a predictive maintenance strategy within its decision-making process. By employing and comparing multiple classifiers, such as logistic regression with Lasso, KNN, random forest and boosting, alongside several rebalance methods, including SMOTE, undersampling techniques and moving threshold approach, a predictive model focused on a feature closely related to maintenance activities has been developed. The analysis aims to accurately estimate the probability that the majority of alerts on a given day would be classified as critical, based on limited information about the train’s status. The results obtained illustrate which features influence the criticality of alerts, while also highlighting the strengths and weaknesses of the various statistical learner and rebalance methods employed.

### Contents

:open_file_folder: Codici_diagnostici --> code to link information on the criticality and severity of alerts with the dataset of recorded alerts during train services (log dataset)

:open_file_folder: Confini_amministrativi_ISTAT --> ISTAT administrative boundaries of municipalities, provinces, and regions of Italy

:open_file_folder: TSR_040 --> various codes to perform the statistical analysis (cleaning, preprocessing, exploratory data analysis and statistical modelling)

:open_file_folder: csv_dataset_TSR --> csv file of log data from TSR trains

:open_file_folder: csv_maintenances_services --> data on scheduled and corrective maintenances, and services performed by trains
