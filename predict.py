from sklearn import preprocessing
from sklearn.preprocessing import scale 
from xgboost import XGBRegressor
import pandas as pd
import requests
import matplotlib.pyplot as plt
import io
import numpy as np
import os
from train_models import train
import sys


def main():

    print("Loading data...")

    # Download the CSV using requests
    url = "https://data.transportation.gov/api/views/keg4-3bc2/rows.csv?accessType=DOWNLOAD"
    response = requests.get(url)

    # Load the CSV content into a pandas DataFrame
    df = pd.read_csv(io.StringIO(response.text))

    df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')

    # Create new columns
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    df = df.drop(['Point'], axis=1)

    # convert all values to lower case to eliminate casing issues
    df = df.map(lambda x: x.lower() if type(x) == str else x)

    port_code = int(sys.argv[1])
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    measure = sys.argv[4].strip().lower()

    if year < 2025 or year > 2027:
        print(f"Invalid year. Must be between 2025 and 2027, inclusive.")
        return
    if month < 1 or month > 12:
        print(f"Invalid month. Must be between 1 and 12, inclusive.")
        return
    if measure not in ["pedestrians", "personal vehicles", "trucks"]:
        print('Invalid measure. Acceptable: "pedestrians", "personal vehicles", "trucks"')
        return

    try:
        port_name = list(df[df["Port Code"] == port_code]['Port Name'])[0]
        state = list(df[df["Port Code"] == port_code]['State'])[0]
        lat = float(list(df[df["Port Code"] == port_code]['Latitude'])[0])
        long = float(list(df[df["Port Code"] == port_code]['Longitude'])[0])
        border = list(df[df["Port Code"] == port_code]['Border'])[0]

        if measure == "pedestrians" and border == "us-canada border":
            print("Invalid input. Pedestrian predictions only supported for US-Mexico ports.")
            return 
        
    except IndexError:
        print("Invalid port code.")
        return
    

    if not os.path.exists("models/XGBR_Can_Tru.json"):
        print("Training models...")
        train(df)

    print("Predicting...")
    # Generate dates and adjust to the 1st of the month
    dates = pd.date_range(start=f"{year}-{month}-01", periods=3, freq="12MS") + pd.offsets.Day(-1)
    dates = dates + pd.offsets.MonthBegin(+1) 

    # Create the data
    data = {
        "Port Name": [port_name] * 3,
        "State": [state] * 3,
        "Port Code": [port_code] * 3,
        "Border": [border] * 3,
        "Date": dates,
        "Measure": [measure] * 3,
        "Latitude": [lat] * 3,
        "Longitude": [long] * 3,
        "month": [month] * 3,
        "year": [2025, 2026, 2027],
    }

    # Create the DataFrame
    df_new = pd.DataFrame(data)

    df_new = df_new.drop(['Port Name', 'Border', 'Measure', 'Date'], axis=1)

    # Label encoding on all non-numeric features
    le_new = preprocessing.LabelEncoder()

    # Select non-numeric features
    cols_obj = [col for col in df_new.columns if df_new[col].dtype in ['object']]

    for col in cols_obj:
        df_new[col] = le_new.fit_transform(df_new[col])

    model = XGBRegressor()

    # print(measure, border)
    if measure == 'pedestrians':
        df_new = pd.DataFrame(scale(df_new, with_std=True))
        model.load_model("models/XGBR_Mex_Ped.json")
        y_pred = model.predict(df_new)

    elif measure == "personal vehicles" and border == "us-canada border":
        df_new = pd.DataFrame(scale(df_new, axis=1, with_std=False))
        model.load_model("models/XGBR_Can_Veh.json")
        y_pred = model.predict(df_new)

    elif measure == "personal vehicles" and border == "us-mexico broder":
        df_new = pd.DataFrame(scale(df_new, axis=1, with_std=False))
        model.load_model("models/XGBR_Mex_Veh.json")
        y_pred = model.predict(df_new)

    elif measure == "trucks" and border == "us-canada border":
        df_new = pd.DataFrame(scale(df_new, axis=1, with_std=False))
        model.load_model("models/XGBR_Can_Tru.json")
        y_pred = model.predict(df_new)

    elif measure == "trucks" and border == "us-mexico border":
        df_new = pd.DataFrame(scale(df_new, axis=1, with_std=False))
        model.load_model("models/XGBR_Mex_Tru.json")
        y_pred = model.predict(df_new)


    df_0 = df.sort_values(by=['year'])

    x = np.asarray(df_0[(df_0['Measure'] == measure) & (df_0['Port Code'] == port_code) & (df_0['month'] == month)]['year'])
    y = np.asarray(df_0[(df_0['Measure'] == measure) & (df_0['Port Code'] == port_code) & (df_0['month'] == month)]['Value'])

    month_dict = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    print(f"The predicted {measure} crossings at Port {port_name} ({port_code}) in {month_dict[month]} {year} is {round(y_pred[year % 2025])}")

    plt.plot(x, y, label='Data', marker='o')
    plt.plot([2025, 2026, 2027], y_pred, linestyle='-.', marker='x', label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('Crossings')
    plt.title(f'{month_dict[month]} {measure} | {port_name} ({port_code})')
    plt.legend()

    # Save the plot
    plt.savefig('Regression Forcast.png')


if __name__ == "__main__":
    main()