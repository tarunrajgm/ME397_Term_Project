from sklearn import preprocessing
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import requests
import io
import numpy as np
import os

def train(df=None):
    if not os.path.exists("models"):
        os.makedirs("models")

    if df is None:
        url = "https://data.transportation.gov/api/views/keg4-3bc2/rows.csv?accessType=DOWNLOAD"
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))

        df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year

        df = df.drop(['Point'], axis=1)

        # convert all values to lower case to eliminate casing issues
        df = df.map(lambda x: x.lower() if type(x) == str else x)

    # MEXICO PEDESTRIANS
    df_pedestrians = df[(df['Measure'] == 'pedestrians') & (df['Border'].str.contains('mexico'))]  

    # Feature Selection
    # Port Name redundant to Port Code, Date redundant to Month and Year
    df_pedestrians_ports = df_pedestrians.drop(['Port Name', 'Border', 'Measure', 'Date'], axis=1)

    # Select non-numeric features (i.e. State) and encode them numerically 
    cols_obj = [col for col in df_pedestrians_ports.columns if df_pedestrians_ports[col].dtype in ['object']]

    le = preprocessing.LabelEncoder()
    for col in cols_obj:
        df_pedestrians_ports[col] = le.fit_transform(df_pedestrians_ports[col])

    X = df_pedestrians_ports.drop(['Value'], axis=1)
    # Scale input features 
    X = pd.DataFrame(scale(X, with_std=False))

    y = df_pedestrians_ports.loc[:,['Value']]

    # Split dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    XGBR_Mex_Ped = XGBRegressor(n_estimators = 1200, learning_rate = 0.14, max_depth = 7, subsample = 1.0)
    XGBR_Mex_Ped.fit(X_train,y_train)

    y_test_pred = XGBR_Mex_Ped.predict(X_test)
    mae  = mean_absolute_error(y_test, y_test_pred)
    mean = np.mean(df_pedestrians_ports['Value'])

    print(f"Percentage Error Mexico Pedestrians Model: {mae/mean*100:.2f}%")

    # path to save the model
    save_path = "models/XGBR_Mex_Ped.json"
    XGBR_Mex_Ped.save_model(save_path)


    # MEXICO VEHICLES
    df_vehicles = df[((df['Measure'] == 'personal vehicles')) & (df['Border'].str.contains('mexico'))]

    # Port Name redundant to Port Code, Date redundant to Month and Year
    df_mex_ports = df_vehicles.drop(['Port Name', 'Measure', 'Border', 'Date'], axis=1)

    # Label encoding on all non-numeric features
    le = preprocessing.LabelEncoder()

    # Select non-numeric features
    cols_obj = [col for col in df_mex_ports.columns if df_mex_ports[col].dtype in ['object']]

    for col in cols_obj:
        df_mex_ports[col] = le.fit_transform(df_mex_ports[col])

    df_mex_ports.head()

    X = df_mex_ports.drop(['Value'], axis=1)
    X = pd.DataFrame(scale(X, axis=1, with_std=False))
    y = df_mex_ports.loc[:,['Value']]

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    XGBR_Mex_Veh = XGBRegressor(n_estimators = 1200, learning_rate = 0.14, max_depth = 7, subsample = 1.0)
    XGBR_Mex_Veh.fit(X_train,y_train)
    # Specify the path to save the model
    save_path = "models/XGBR_Mex_Veh.json"

    # Save the model
    XGBR_Mex_Veh.save_model(save_path)

    mean = np.mean(df_mex_ports['Value'])
    y_test_pred = XGBR_Mex_Veh.predict(X_test)
    mae  = mean_absolute_error(y_test, y_test_pred)

    print(f"Percentage Error Mexico Vehicles Model: {mae/mean*100:.2f}%")


    # MEXICO TRUCKS
    df_trucks_mex = df[((df['Measure'].str.contains('trucks'))) & (df['Border'].str.contains('mexico'))] 

    # Port Name redundant to Port Code, Date redundant to Month and Year
    df_mex_ports = df_trucks_mex.drop(['Port Name', 'Measure', 'Border', 'Date'], axis=1)

    # Label encoding on all non-numeric features
    le = preprocessing.LabelEncoder()

    # Select non-numeric features
    cols_obj = [col for col in df_mex_ports.columns if df_mex_ports[col].dtype in ['object']]

    for col in cols_obj:
        df_mex_ports[col] = le.fit_transform(df_mex_ports[col])

    df_mex_ports.head()

    X = df_mex_ports.drop(['Value'], axis=1)
    X = pd.DataFrame(scale(X, axis=1, with_std=False))
    y = df_mex_ports.loc[:,['Value']]

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    XGBR_Mex_Tru = XGBRegressor(n_estimators = 1200, learning_rate = 0.14, max_depth = 7, subsample = 1.0)
    XGBR_Mex_Tru.fit(X_train, y_train)
    # Specify the path to save the model
    save_path = "models/XGBR_Mex_Tru.json"

    # Save the model
    XGBR_Mex_Tru.save_model(save_path)

    mean = np.mean(df_mex_ports['Value'])

    y_test_pred = XGBR_Mex_Tru.predict(X_test)
    mae  = mean_absolute_error(y_test, y_test_pred)

    print(f"Percentage Error Mexico Trucks Model: {mae/mean*100:.2f}%")


    # CANADA TRUCKS
    df_trucks = df[((df['Measure'].str.contains('trucks'))) & (df['Border'].str.contains('canada'))] 

    # Port Name redundant to Port Code, Date redundant to Month and Year
    df_can_ports = df_trucks.drop(['Port Name', 'Measure', 'Border', 'Date'], axis=1)

    # Label encoding on all non-numeric features
    le = preprocessing.LabelEncoder()

    # Select non-numeric features
    cols_obj = [col for col in df_can_ports.columns if df_can_ports[col].dtype in ['object']]

    for col in cols_obj:
        df_can_ports[col] = le.fit_transform(df_can_ports[col])

    df_can_ports.head()

    X = df_can_ports.drop(['Value'], axis=1)
    X = pd.DataFrame(scale(X, axis=1, with_std=False))
    y = df_can_ports.loc[:,['Value']]

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    XGBR_Can_Tru = XGBRegressor(n_estimators = 1200)
    XGBR_Can_Tru.fit(X_train, y_train)
    # Specify the path to save the model
    save_path = "models/XGBR_Can_Tru.json"

    # Save the model
    XGBR_Can_Tru.save_model(save_path)

    mean = np.mean(df_can_ports['Value'])

    y_test_pred = XGBR_Can_Tru.predict(X_test)
    mae  = mean_absolute_error(y_test, y_test_pred)

    print(f"Percentage Error Canada Trucks Model: {mae/mean*100:.2f}%")


    # CANADA VEHICLES
    df_vehicles = df[(df['Measure'] == 'personal vehicles') & df['Border'].str.contains('canada')] # US-Canada border vehicles

    # Port Name redundant to Port Code, Date redundant to Month and Year
    df_can_ports = df_vehicles.drop(['Port Name', 'Measure', 'Border', 'Date'], axis=1)

    # Label encoding on all non-numeric features
    le = preprocessing.LabelEncoder()

    # Select non-numeric features
    cols_obj = [col for col in df_can_ports.columns if df_can_ports[col].dtype in ['object']]

    for col in cols_obj:
        df_can_ports[col] = le.fit_transform(df_can_ports[col])

    df_can_ports.head()

    X = df_can_ports.drop(['Value'], axis=1)
    X = pd.DataFrame(scale(X, axis=1, with_std=False))
    y = df_can_ports.loc[:,['Value']]

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    XGBR_Can_Veh = XGBRegressor(n_estimators=1400)
    XGBR_Can_Veh.fit(X_train,y_train)
    # Specify the path to save the model
    save_path = "models/XGBR_Can_Veh.json"

    # Save the model
    XGBR_Can_Veh.save_model(save_path)

    mean = np.mean(df_can_ports['Value'])

    y_test_pred = XGBR_Can_Veh.predict(X_test)
    mae  = mean_absolute_error(y_test, y_test_pred)

    print(f"Percentage Error Canada Vehicles Model: {mae/mean*100:.2f}%")


if __name__ == "__main__":
    train()