from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def my_load_boston():
    boston_data = load_boston()

    X = boston_data["data"]
    y = boston_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = my_load_boston()
    print("X_train head: {}".format(X_train[:5]))
    print("X_test head: {}".format(X_test[:5]))
    print("y_train head: {}".format(y_train[:5]))
    print("y_test head: {}".format(y_test[:5]))
    print("features: 13")