# Classification Metrics
SVMaccuracy  = accuracy_score(Y_test, SVMclassifier.predict(X_test[:,:feature_limit]))
SVMprecision = precision_score(Y_test, SVMclassifier.predict(X_test[:,:feature_limit]),  average="macro")
SVMrecall    = recall_score(Y_test, SVMclassifier.predict(X_test[:,:feature_limit]),  average="macro")
SVMF1        = f1_score(Y_test, SVMclassifier.predict(X_test[:,:feature_limit]),  average="macro")

RFaccuracy  = accuracy_score(Y_test, RFclassifier.predict(X_test[:,:feature_limit]))
RFprecision = precision_score(Y_test, RFclassifier.predict(X_test[:,:feature_limit]),  average="macro")
RFrecall    = recall_score(Y_test, RFclassifier.predict(X_test[:,:feature_limit]),  average="macro")
RFF1        = f1_score(Y_test, RFclassifier.predict(X_test[:,:feature_limit]),  average="macro")

print("Super Vector Machine")
print("Accuracy = %0.4f"%SVMaccuracy), print("Precision = %0.4f"%SVMprecision)
print("Recall = %0.4f"%SVMrecall),     print("F1 Score = %0.4f"%SVMF1)
print("---------------------------------")
print("Random Forest")
print("Accuracy = %0.4f"%RFaccuracy),  print("Precision = %0.4f"%RFprecision)
print("Recall = %0.4f"%RFrecall),      
print("F1 Score = %0.4f"%RFF1)