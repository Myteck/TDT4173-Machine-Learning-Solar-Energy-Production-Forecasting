
# Machine Learning Model
import catboost as cb

# Data Processing Tools
import pandas as pd

# Machine Learning Tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import data_pipeline as dp
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
 
class learner:
  def __init__(self, file_paths: list[list[str]], features: list[str] = [], learning_algorithm: str = cb.CatBoostRegressor) -> None:
    self.files = files
    self.features = features
    self.learning_algorithm = learning_algorithm
    self.buildings = ["A", "B", "C"]

def create_training_data(self):
    list_y = []
    list_X = []
    list_X_pred = []
   
    for path, i in enumerat(self.file_paths):
        y = pd.read_parquet(path[0])
        X_estimated = pd.read_parquet(path[1])
        X_observed = pd.read_parquet(path[2])
        X_pred = pd.read_parquet(path[3])

        # =================  TEST DATA  ================
        X_pred = dp.pred_data_processing(X_pred)
        X_pred['building'] = self.buildings[i]
        list_X_pred.append(X_pred)

        # =================TRAINING DATA================
        # Pre-process data
        y = y.dropna()
        X_estimated = X_estimated.drop("date_calc", axis = 1)
        X = pd.concat([X_observed, X_estimated], axis = 0, ignore_index=True)
        
        # BETTER NAME
        X, y= dp.train_data_processing(X, y)
        
        # ADD A FUNCTION TO GENERATE BUILDING FEATURE.
        X['building'] = self.buildings[i]

        # Adding the datasets to the lists
        list_y.append(y)
        list_X.append(X)
    
    # Add all the lists together. However there is a need to add set
    y = pd.concat(list_y, axis="rows")
    X = pd.concat(list_X, axis="rows")

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

    X_pred = pd.concat(list_X_pred)
    
    # ================= SCALING DATA================
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    self.X_pred = scaler(X_pred)
    return None

def fit_model(self):
    """
    Based on the selected model the class switches between what model is doing the learning. 
    """

    # Add a function that picks between different models, and processes the data based on this
    self.train_dataset = cb.Pool(self.X_train, self.y_train)
    self.test_dataset = cb.Pool(self.X_test, self.y_test)

    self.model = cb.CatBoostRegressor(loss_function="MAE", logging_level='Silent')

    grid = {'iterations': [100, 150, 200],
            'learning_rate': [0.03, 0.1],
            'depth': [2, 4, 6, 8],
            'l2_leaf_reg': [0.2, 0.5, 1, 3]}

    self.model.grid_search(grid, train_dataset, verbose=False)
    
    return None

def get_performance(self) -> None:
    pred = self.model.predict(self.X_train)
    mae = (mean_absolute_error(y_test, pred))
    print("Mean Abs: {:.2f}".format(mae))

def generate_predictions(self) -> None:
    unformated_pred = self.model.predict(self.X_train)
    pred = self._format_predictions(unformated_pred)
    self._save_predictions(pred)
    

def _format_predictions(unformated_pred: pd.DataFrame) -> pd.DataFrame:
    
    # 
    to_be_submitted_index = pd.read_csv("test.csv")

    #convert the "time" column to datetime
    to_be_submitted_index["time"] = pd.to_datetime(to_be_submitted_index["time"])
    pred = pd.merge(unformated_pred, to_be_submitted_index, how='inner', left_on=['date_forecast', 'building'], right_on=["time", "location"])
    print(len(X_test_resampled.index))
    return pred
    
    return None
def _save_predictions(pred: pd.DataFrame)->None:
    #Make the index and pv_measurement column into a csv file
    pred[["id", "pv_measurement"]].rename(columns={"id" : "id" , "pv_measurement" : "prediction"}).to_csv("model_pred.csv", index=False)

def _predict():
    return None






def create_learner(X: pd.DataFrame, y: pd.DataFrame, X_pred: pd.DataFrame):
    # Should be split into several parts
    # We want to merge all the different trainingsets into 1 large \

    #


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
