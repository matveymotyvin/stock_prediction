import sys
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LogisticRegression


def train_forecasting_model(data_path, len_test):
    # Load and preprocess data for forecasting
    df = pd.read_csv(data_path)
    df['дата'] = pd.to_datetime(df['дата'], format='%d.%m.%Y')
    df = df.sort_values(by='дата')
    df['выход'] = df['выход'].str.replace(',', '.').astype(float)
    df['выход'] = df['выход'].astype(float)

    train_forecast = df[['выход', 'дата']]
    train_forecast.set_index('дата', inplace=True)

    # Train SARIMAX model
    order = (0, 1, 0)
    seasonal_order = (0, 1, 0, 12)
    model = SARIMAX(train_forecast, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()

    # Generate forecast
    forecast = result.forecast(steps=len_test)

    with open('forecast_value.json', 'w') as file:
        json.dump(forecast.tolist(), file)
        
    return forecast.tolist()


def train_classification_model(data_path):
    # Load data for classification
    df_class = pd.read_csv(data_path)
    df_class['дата'] = pd.to_datetime(df_class['дата'], format='%d.%m.%Y')
    df_class = df_class.sort_values(by='дата')
    df_class['выход'] = df_class['выход'].str.replace(',', '.').astype(float)
    df_class['выход'] = df_class['выход'].astype(float)
    df_class['направление'] = df_class['направление'].replace({'л': 1, 'ш': 0})
    
    df_class['пред_направление'] = df_class['направление'].shift(1)
    df_class['пред_выход_1'] = df_class['выход'].shift(1)
    df_class['пред_выход_2'] = df_class['выход'].shift(2)
    df_class['пред_выход_3'] = df_class['выход'].shift(3)
    df_class['пред_выход_4'] = df_class['выход'].shift(4)
    df_class['пред_выход_5'] = df_class['выход'].shift(5)

    df_class.dropna(inplace=True)

    X_train = df_class[['выход', 'пред_направление', 'пред_выход_5', 'пред_выход_4', 'пред_выход_3',
                        'пред_выход_2', 'пред_выход_1']]
    y_train = df_class['направление']

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def generate_classification_forecast(model, forecast, last_five_pred, last_direction, len_test):
    class_pred = []

    for i in range(len_test):
        feature_vector = np.array([list([forecast[i]]) + [last_direction] + last_five_pred])
        pred = model.predict(feature_vector)
        class_pred.append(int(pred[0]))
        last_direction = pred[0]
        last_five_pred.pop(0)
        last_five_pred.append(forecast[i])

    with open('forecast_class.json', 'w') as file:
        json.dump(class_pred, file)


def main(data_path='data.csv', len_test=48):
    # Train forecasting model and get forecast
    forecast = train_forecasting_model(data_path, len_test)

    # Train classification model
    model = train_classification_model(data_path)

    df = pd.read_csv(data_path)
    df['дата'] = pd.to_datetime(df['дата'], format='%d.%m.%Y')
    df = df.sort_values(by='дата')
    df['выход'] = df['выход'].str.replace(',', '.').astype(float)
    df['выход'] = df['выход'].astype(float)
    df['направление'] = df['направление'].replace({'л': 1, 'ш': 0})
    
    last_five_pred = df[-5:]['выход'].values.tolist()
    last_direction = df['направление'].values[-1]

    generate_classification_forecast(model, forecast, last_five_pred, last_direction, len_test)

    print('Pipeline completed successfully!')


if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) == 3:
        data_path = sys.argv[1]
        len_test = int(sys.argv[2])
        main(data_path, len_test)
    else:
        # Use default values if no arguments are provided
        main()
