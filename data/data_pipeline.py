"""
This library is created to simplify testing. There is a lot of dataprocessing going on, and changing
how features are extracted and filtered is essential for easing the whole development process.
"""
import feature_generation as feat_gen
import data_processing as dat_proc
import pandas as pd

def train_data_processing(X: pd.DataFrame, y: pd.DataFrame, filter_list: list[str] = [], add_y_signal: bool = False):
   
    # Removing NaN values. If there are missing values treat start and end points as beginning and end of a line.
    X = X.interpolate(method='linear', limit_direction = "both")

    # Extract necesarry values for feature generation.
    timestamps = "date_forecast"
    measurements = list(X.columns.values)
    measurements.remove(timestamps)

    # Probable features that may be used
    der_df = feat_gen.difference_df(X, timestamps, measurements)
    dder_df = feat_gen.double_derivative_from_df(X, timestamps, measurements)
    int_df = feat_gen.daily_accumulated_val_df(X, timestamps, measurements)
    dint_df = feat_gen.daily_accumulated_val_squared_df(X, timestamps, measurements)
    time_df = feat_gen.time_data_from_df(X, timestamps)

    X = pd.concat([X, der_df, dder_df, dint_df, int_df, time_df], axis = "columns")

    if len(filter_list) > 0:
        X = X[filter_list + ["date_forecast"]]

    # Additional features
    der_y = feat_gen.difference_df(y, "time", ["pv_measurement"])
    der_y_shifted = feat_gen.shifted_values_24_h(der_y, "derivative_pv_measurement_d")
    y_shifted =  feat_gen.shifted_values_24_h(y, "pv_measurement")

    # Adding together the added features to one dataframe.
    y_BIG = pd.concat([y, der_y_shifted, y_shifted])


    # Making sure that the two dataframes match in length.
    y_BIG, X = dat_proc.data_length_matching(y_BIG, X)

    # Get our desired output
    y = y_BIG["pv_measurement"]
    y = y.reset_index(drop = True)
    
    
    if add_y_signal:
        # Removing datetime object column.
        y_features = y_BIG.drop('pv_measurement', axis=1)
        y_features = y_features.drop('time', axis=1)
        y_features = y_features.reset_index(drop = True)

        X = pd.concat([X, y_features], axis = 1)
    
    # Removing datetime object column
    X = X.drop(timestamps, axis=1)
    X = X.reset_index(drop = True)

    return X, y

def pred_data_processing(X_pred: pd.DataFrame, filter_list: list[str] = []) -> pd.DataFrame:
    """
    A function that reads
    """
    
    # Removing NaN values. If there are missing values treat start and end points as beginning and end of a line.
    X_pred = X_pred.interpolate(method = 'linear')
    X_pred = X_pred.bfill()

    # Extract necesarry values for feature generation.
    timestamps = "date_forecast"

    # Removing date-time from measurements
    measurements = list(X_pred.columns.values)
    measurements.remove("date_forecast")
    measurements.remove("date_calc")

    # Probable features that may be used
    der_df = feat_gen.difference_df(X_pred, timestamps, measurements)
    dder_df = feat_gen.double_derivative_from_df(X_pred, timestamps, measurements)
    int_df = feat_gen.daily_accumulated_val_df(X_pred, timestamps, measurements)
    dint_df = feat_gen.daily_accumulated_val_squared_df(X_pred, timestamps, measurements)
    time_df = feat_gen.time_data_from_df(X_pred, timestamps)

    

    X_pred_new = pd.concat([X_pred, der_df, dder_df, dint_df, int_df, time_df], axis = "columns")

    if len(filter_list) > 0:
        X_pred_new = X_pred_new[filter_list + ['date_forecast']]
        
    else:
        X_pred_new = X_pred_new.drop("date_calc", axis = 1)

    return X_pred_new

