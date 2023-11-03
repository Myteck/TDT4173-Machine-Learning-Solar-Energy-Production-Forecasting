
# Machine Learning Model
import catboost as cb

# Data Processing Tools
import pandas as pd

# Machine Learning Tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import data_pipeline as dp
from sklearn.preprocessing import MinMaxScaler

def create_learner(X: pd.DataFrame, y: pd.DataFrame, X_pred: pd.DataFrame):
    X, y= dp.train_data_processing(X, y)

    # DO we get a day or the 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

    scaler = MinMaxScaler()

    # Fit and transform the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = cb.Pool(X_train, y_train) 
    test_dataset = cb.Pool(X_test, y_test)

    model = cb.CatBoostRegressor(loss_function="MAE", logging_level='Silent')

    grid = {'iterations': [100, 150, 200],
            'learning_rate': [0.03, 0.1],
            'depth': [2, 4, 6, 8],
            'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    model.grid_search(grid, train_dataset, verbose=False)


    pred = model.predict(X_test)
    mae = (mean_absolute_error(y_test, pred))
    print("Testing performance")
    print("Mean Abs: {:.2f}".format(mae))

    N = 100

    feature_importance = model.get_feature_importance()

    # Pair feature names with their importance scores
    feature_importance_dict = dict(zip(model.feature_names_, feature_importance))

    # Sort features by importance
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print or use the top features
    top_features = sorted_feature_importance[:N]  # Replace N with the number of top features you want
    print(top_features)
    labels = list(X.columns.values)

    best_features = []
    for feat in top_features:
        print(labels[int(feat[0])])
        best_features.append(labels[int(feat[0])])
    print(best_features)
    
    
    X_pred = dp.pred_data_processing(X_pred)

    X_val = scaler.transform(X_pred)
    pred = model.predict(X_val)
    pred_df = pd.DataFrame(pred)
    pred_df.to_csv('A_Pred.csv')

    return model
