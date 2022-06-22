

# precision    recall  f1-score   support
from sklearn.metrics import classification_report
y_true = ['y', 'y', 'y', 'n', 'n']
y_pred = ['y', 'n', 'y', 'n', 'n']
target_names = ['y', 'n']
print(classification_report(y_true, y_pred, target_names=target_names))


from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))

