from datetime import datetime, timedelta

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