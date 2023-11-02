"""
This is a helper library to generate features.
"""
# Data Processing Tools
import numpy as np
import pandas as pd

def difference_df(df: pd.DataFrame, timeStamps: str, measurements: list[str]) -> pd.DataFrame:
    """
    Adds a derivative column to the pandas dataframe. May be used to create time dependency.
    """
    der_df = pd.DataFrame()
    df = df.copy()
    # Sort DataFrame by timestamp
    df = df.sort_values(timeStamps) 

    # Calculate time differences
    df['time_diff'] = df[timeStamps].diff()

    for measurement in measurements:
        # Calculate derivative estimates
        der_df['derivative_' + measurement + '_d'] = df[measurement].diff()
    
    df = df.drop('time_diff', axis =  1)

    # Since the first element will result in a NaN, we must backfill this one.
    der_df = der_df.interpolate(method='linear')
    der_df = der_df.bfill()
    
    return der_df

def double_derivative_from_df(df: pd.DataFrame, timeStamps: str, measurements: list[str]) -> pd.DataFrame:
    """
    Adds a derivative column to the pandas dataframe. May be used to create time dependency.
    """
    dder_df = pd.DataFrame()
    df = df.copy()
    # Sort DataFrame by timestamp
    df = df.sort_values(timeStamps) 

    # Calculate time differences
    df['time_diff'] = df[timeStamps].diff()

    # Calculate derivative estimates
    for measurement in measurements:
        dder_df['double_derivative_' + measurement + '_dd'] = df[measurement].diff() / (divmod(df['time_diff'].dt.total_seconds(), 60)[0]**2)
    
    df = df.drop('time_diff', axis=1)
    
    # Since the first element will result in a NaN, we must backfill this one.
    dder_df = dder_df.interpolate(method='linear')
    dder_df = dder_df.fillna(method="backfill", axis=None)

    return dder_df

def daily_accumulated_val_df(df: pd.DataFrame, timeStamps: str, measurements: list[str]) -> pd.DataFrame:
    
    i_df = pd.DataFrame()
    df = df.copy()
    # Sort DataFrame by timestamp
    df = df.sort_values(timeStamps)

    # Create a new column for the date
    df['date'] = df[timeStamps].dt.date

    for measurement in measurements:
        # Calculate the integral value for each day
        i_df['integral_' + measurement + '_integral'] = df.groupby('date')[measurement].cumsum()
    
    df = df.drop('date', axis=1)

    return i_df

def daily_accumulated_val_squared_df(df: pd.DataFrame, timeStamps: str, measurements: list[str]) -> pd.DataFrame:
    
    di_df = pd.DataFrame()
    df = df.copy()
    # Sort DataFrame by timestamp
    df = df.sort_values(timeStamps)

    # Create a new column for the date
    df['date'] = df[timeStamps].dt.date

    for measurement in measurements:
        # Calculate the integral value for each day
        di_df['double_integral_' + measurement + '_dintegral'] = df.groupby('date')[measurement].cumsum()**2
    
    df = df.drop('date', axis=1)

    return di_df

def time_data_from_df(df: pd.DataFrame, timestamps: str) -> pd.DataFrame: 
    # Extracting components
    time_df = pd.DataFrame()
    df = df.copy()
    time_df['day_of_year:day'] = df[timestamps].dt.dayofyear
    time_df['month:month'] = df[timestamps].dt.month
    #time_df['year:year'] = df[timestamps].dt.year
    time_df['hour:hour'] = df[timestamps].dt.hour
    return time_df



def n_largest_freq(df: pd.DataFrame, measurements: list[str],n_largest: int):
    """
    Generates values based on the largest frequencies that are present.
    """
    for measurement in measurements:
        signal = df[measurement].values

        fft_result = np.fft.fft(signal)
        
        # Keep only the dominant frequencies (e.g., top 5)
        num_components_to_keep = n_largest

        indices = np.argsort(np.abs(fft_result))[::-1][:num_components_to_keep]

        # Set all other frequency components to zero
        fft_result_filtered = np.zeros_like(fft_result)
        fft_result_filtered[indices] = fft_result[indices]

        # Compute IFFT
        ifft_result = np.fft.ifft(fft_result_filtered)

        # Add the filtered results to the dataframe
        df["filtered_" + measurement] = ifft_result.real

    return df

def freq_comb(df: pd.DataFrame, features: list[str]) -> np.array:
    """
    Takes the fourier transform of multiple signals add them together, and then takes the inverse.

    features: Are what you would like to combine.
    df: Chosen dataframe containing feature information.
    """

    total_fft = 0
    
    for feat in features:
        # Finding the signal directly might be wrong due to timestamps and such, but might still be helpful. It is not correct, but improvements like day by day sampling might be useful.
        signal = df[feat].values

        # Min-max scaling
        scaled_signal = min_max_scale(signal)
        
        fft = np.fft.fft(scaled_signal)
        total_fft = total_fft + fft
    
    ifft_result = np.fft.ifft(total_fft)

    return ifft_result.real

def min_max_scale(signal: np.array) -> np.array:
    # Calculate min and max values
    min_val = np.min(signal)
    max_val = np.max(signal)

    # Min-max scaling
    scaled_signal = (signal - min_val) / (max_val - min_val)

    return scaled_signal

def shifted_values_24_h(y: pd.DataFrame, measurement: str)->pd.DataFrame:
    df = pd.DataFrame()
    for i in range(1, 25):
        df[measurement + 'n-' + str(i)] = y[measurement].shift(i)
    
    return df

def merge_features(df: pd.DataFrame):
    # Extract the part before ":" in column names
    df.columns = df.columns.str.split(':').str[0]

    # Group by modified column names and sum values
    grouped_df = df.groupby(df.columns, axis=1).sum()

    return grouped_df