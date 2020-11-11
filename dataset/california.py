from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def load_california(split = True, debug = False, normalize = True):
    california_data = fetch_california_housing(data_home = "./")

    features = california_data["feature_names"]
    X = california_data["data"]
    y = california_data["target"]

    
    X_df = pd.DataFrame(X, columns = features)
    y_df = pd.DataFrame(y, columns = ["HousePrice"])
    data = pd.concat([X_df, y_df], axis = 1)
    if(debug):
        print("ABOUT DATA")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print(california_data["DESCR"])
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

        print("DATA")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        X_df = pd.DataFrame(X, columns = features)
        y_df = pd.DataFrame(y, columns = ["HousePrice"])
        data = pd.concat([X_df, y_df], axis = 1)
        print("1. data")
        print(data)
        print("\n2. statistical description of data")
        print(data.describe())
        print("\n3. lack of data")
        print(data.isnull().sum())
        N = X.shape[1]
        fig = plt.figure()
        for i in range(N):
            ax = fig.add_subplot((N+2)//3, 3, i + 1)
            ax.scatter(X[:, i], y, marker = ".")
            ax.set_xlabel(features[i])
            ax.set_ylabel("HousePrice")
        plt.show()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    if(normalize):
        x_tmp = X_df.abs()
        
        print(X_df)
        print(x_tmp)
        
        n = [x_tmp["MedInc"].max(), 1, 1, 1, 1, 1, 1, 1]
        X = X/n
        
        n = [1, x_tmp["HouseAge"].max(), 1, 1, 1, 1, 1, 1]
        X = X/n
        
        n = [1, 1, x_tmp["AveRooms"].max(), 1, 1, 1, 1, 1]
        X = X/n
        
        n = [1, 1, 1, x_tmp["AveBedrms"].max(), 1, 1, 1, 1]
        X = X/n
        
        n = [1, 1, 1, 1, x_tmp["Population"].max(), 1, 1, 1]
        X = X/n
        
        n = [1, 1, 1, 1, 1, x_tmp["AveOccup"].max(), 1, 1]
        X = X/n
        
        n = [1, 1, 1, 1, 1, 1, x_tmp["Latitude"].max(), 1]
        X = X/n
        
        n = [1, 1, 1, 1, 1, 1, 1, x_tmp["Longitude"].max()]
        X = X/n
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        return X_train, X_test, y_train, y_test
    
    return california_data

if __name__ == "__main__":
    load_california(debug = True)


