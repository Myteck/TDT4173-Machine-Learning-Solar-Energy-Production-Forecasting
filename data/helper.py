from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

def find_missing_dates(dates: list[datetime], hours: int, mins: int)->list[datetime]: 

    datetime_obj = dates.iloc[0]
    missing_dates = []

    count = 0
    while dates.iloc[count] != dates.iloc[-1]:
        if dates.iloc[count] != datetime_obj:
            missing_dates.append(datetime_obj)
        else:
            count += 1
        datetime_obj += timedelta(hours= hours, minutes = mins)
    return missing_dates

def reformat_time(df_missing_format: pd.DataFrame, df_desired_format: pd.DataFrame) -> pd.DataFrame:
    df_missing_format["date_forecast"] = pd.to_datetime(df_missing_format["date_forecast"])
    df = df_missing_format.groupby([df_missing_format["date_forecast"].dt.hour]).mean()
    df["date_forecast"] = df_desired_format["time"]
    return df

def corr_mat(dataFrame_1: pd.DataFrame, dataFrame_2: pd.DataFrame, title: str)-> None:
    dataFrame_1["pv_measurement"] = dataFrame_2["pv_measurement"]

    corr_matrix = dataFrame_1.corr()

    sns.heatmap(corr_matrix, cmap="PiYG")
    plt.title(title)
    plt.show()