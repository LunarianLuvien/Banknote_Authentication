import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Import the dataset and split into features and dependent variable
dataset = pd.read_csv('data_banknote_authentication.txt', delimiter=',')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split into 80% training and 20% test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Standardize the data after splitting to prevent data leakage
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # Only transform the test data, do not fit

# Train a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict test results with KNN
y_pred_for_knn = knn.predict(x_test)

# Train and predict using an SVM classifier
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
y_pred_for_svm = svm.predict(x_test)

# Predictions for a given sample data
sample_data = [1.14, -3.75, 5.60, -0.64]
transformed_sample_data = sc.transform([sample_data])
svm_prediction_for_given_data = svm.predict(transformed_sample_data)
knn_prediction_for_given_data = knn.predict(transformed_sample_data)
print('Results: 0 indicates fake, 1 indicates genuine.')
print(f'Result from KNN for {sample_data}: ', knn_prediction_for_given_data)
print(f'Result from SVM for {sample_data}: ', svm_prediction_for_given_data)
print()

# Evaluate the KNN model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

cm_knn = confusion_matrix(y_test, y_pred_for_knn)
print("KNN Confusion Matrix:")
print(cm_knn)

plt.matshow(cm_knn)
plt.title('Confusion matrix for KNN model')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

accuracy_scores_of_knn = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10)
mean_of_accuracies_for_knn = accuracy_scores_of_knn.mean()
mean_of_precisions_for_knn = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10, scoring='precision').mean()
mean_of_recall_for_knn = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10, scoring='recall').mean()

print('KNN Model Metrics:')
print(f'Accuracy: {mean_of_accuracies_for_knn * 100:.2f}%')
print(f'Precision: {mean_of_precisions_for_knn * 100:.2f}%')
print(f'Recall: {mean_of_recall_for_knn * 100:.2f}%')
print()

# Evaluate the SVM model
cm_svm = confusion_matrix(y_test, y_pred_for_svm)
print("SVM Confusion Matrix:")
print(cm_svm)

plt.matshow(cm_svm)
plt.title('Confusion matrix for SVM model')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

accuracy_scores_of_svm = cross_val_score(estimator=svm, X=x_train, y=y_train, cv=10)
mean_of_accuracies_for_svm = accuracy_scores_of_svm.mean()
mean_of_precisions_for_svm = cross_val_score(estimator=svm, X=x_train, y=y_train, cv=10, scoring='precision').mean()
mean_of_recall_for_svm = cross_val_score(estimator=svm, X=x_train, y=y_train, cv=10, scoring='recall').mean()

print('SVM Model Metrics:')
print(f'Accuracy: {mean_of_accuracies_for_svm * 100:.2f}%')
print(f'Precision: {mean_of_precisions_for_svm * 100:.2f}%')
print(f'Recall: {mean_of_recall_for_svm * 100:.2f}%')
print()

# Detailed scores for each model
print("Detailed KNN Scores:")
print(f"Accuracy: {[round(score * 100, 2) for score in accuracy_scores_of_knn]}")
print(f"Recall: {[round(score * 100, 2) for score in recall_scores_for_knn]}")
print(f"Precision: {[round(score * 100,
