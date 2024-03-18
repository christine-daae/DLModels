from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from Datasets.LoadData3 import Get_data

def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    # Create an SVM classifier
    clf = svm.SVC(kernel='rbf', random_state=42, max_iter=1000, verbose=1, C = 1.0)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Assuming you already have X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = Get_data(15)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# 使用StandardScaler对数据进行标准化
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("data load finished!")
# Call the function to train and evaluate SVM
accuracy = train_and_evaluate_svm(X_train, X_test, y_train, y_test)

# Print the accuracy
print("Accuracy:", accuracy)
