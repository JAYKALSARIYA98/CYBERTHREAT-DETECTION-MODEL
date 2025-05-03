from sklearn.metrics import accuracy_score

# Print accuracy of binary classification model
print("Binary Model Accuracy:", accuracy_score(y_bin_test, binary_preds))

# Print accuracy of multiclass classification model
print("Multiclass Model Accuracy:", accuracy_score(y_cat_test, multiclass_preds))
