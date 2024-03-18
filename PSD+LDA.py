import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from Datasets.LoadData3 import Get_data



def train_and_evaluate_lda(X_train, X_test, y_train, y_test):
    # Initialize the LDA model
    lda = LinearDiscriminantAnalysis()

    # Train the LDA model
    lda.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lda.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = Get_data(14)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print("data load finished!")


# Train and evaluate the LDA model
lda_accuracy = train_and_evaluate_lda(X_train, X_test, y_train, y_test)

# Print the accuracy
print(f'Accuracy: {lda_accuracy:.2f}')
