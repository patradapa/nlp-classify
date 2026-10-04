from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from classical_ml.preprocessing import (
    X_train_tfidf, X_test_tfidf, y_train, y_test,
)

svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)
print("SVM:\n", classification_report(y_test, svm_preds))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)
print("Random Forest:\n", classification_report(y_test, rf_preds))
