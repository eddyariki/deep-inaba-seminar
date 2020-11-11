import numpy as np
from sklearn.model_selection import train_test_split

def load_easy_data(size = 500, dim = 1):
    X = []
    for i in range(dim):
        X.append(np.random.rand(size))
        
    X = np.array(X).T
    
    y = np.sum(X, axis = 1) / np.sqrt(dim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    return X_train, X_test, y_train, y_test


def load_easy_data_sin(size=500,range_pi=4*np.pi,dim=1):
    X = []
    for i in range(dim):
        X.append(np.linspace(-range_pi,range_pi,num=size))
        
    X = np.array(X).T
    
    y = np.sin(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    return X_train, X_test, y_train, y_test




if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_easy_data()
    print(X_train[:5])
    print(X_test[:5])
    print(y_train[:5])
    print(y_test[:5])
