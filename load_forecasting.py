import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from pandas.api.types import CategoricalDtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Flatten, Dense
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import joblib
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

class TabularLSTMModel:
    def __init__(self, input_shape, lstm_units=[64, 32], output_units=1, model_path=None):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.output_units = output_units
        if model_path is None:
            self.model = self.build_model()
        else:
            self.model = joblib.load(model_path)

    def build_model(self):
        model = Sequential()
        for units in self.lstm_units:
            model.add(LSTM(units, return_sequences=True, input_shape=self.input_shape, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.output_units))
        return model

    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def fit(self, X_train, y_train, epochs=10, batch_size=32,verbose=1):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, data):
        return self.model.predict(data)

    def summary(self):
        return self.model.summary()

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    @classmethod
    def load_model(cls, model_path):
        return cls(None, model_path=model_path)

class TabularGRUModel:
    def __init__(self, input_shape, gru_units=[64, 32], output_units=1):
        self.input_shape = input_shape
        self.gru_units = gru_units
        self.output_units = output_units
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        for units in self.gru_units:
            model.add(GRU(units, return_sequences=True, input_shape=self.input_shape, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.output_units))
        return model

    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        loss = MeanSquaredError()
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, data):
        return self.model.predict(data)
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)


class RegressionModels:
    def __init__(self):
        pass
    
    def train_regression_models(self, X_train, y_train):
        models = [
            ("Linear Regression", LinearRegression()),
            ("Decision Tree", DecisionTreeRegressor()),
            ("Random Forest", RandomForestRegressor()),
            ("Gradient Boosting", GradientBoostingRegressor()),
            ("SVR", SVR()),
            ("KNN", KNeighborsRegressor()),
        ]
        trained_models = {}
        for name, model in models:
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models

    def make_predictions(self, trained_models, X_test):
        predictions = {}
        for name, model in trained_models.items():
            predictions[name] = model.predict(X_test)
        return predictions




def main():
    st.title("Load Forecasting")
    df=pd.read_csv("models\continuous dataset.csv")

    datetime = st.text_input("Enter datetime (YYYY-MM-DD HH:MM:SS):")
    # Other input fields can be added here
    t2m_toc = st.number_input("Enter T2M_toc:")
    qv2m_toc = st.number_input("Enter QV2M_toc:")
    TQL_toc = st.number_input("Enter TQL_toc:")
    W2M_toc= st.number_input("Enter W2M_toc:")
    T2M_san = st.number_input("Enter T2M_san:")
    QV2M_san = st.number_input("Enter QV2M_san:")
    TQL_san= st.number_input("Enter TQL_san:")
    W2M_san = st.number_input("Enter W2M_san:")
    T2M_dav = st.number_input("Enter T2M_dav:")
    QV2M_dav = st.number_input("Enter QV2M_dav:")
    TQL_dav = st.number_input("Enter TQL_dav:")
    W2M_dav = st.number_input("Enter W2M_dav:")
    Holiday_ID = st.number_input("Enter Holiday_ID:")
    holiday = st.number_input("Enter holiday:")
    school = st.number_input("Enter  school:")
    
    if st.button("Submit"):

        sample_input_data = {
            'datetime': [datetime],
            'nat_demand':[0.0],
            'T2M_toc': [t2m_toc],
            'QV2M_toc': [qv2m_toc],
            'TQL_toc': [TQL_toc],
            'W2M_toc': [W2M_toc],
            'T2M_san': [T2M_san],
            'QV2M_san': [QV2M_san],
            'TQL_san': [TQL_san],
            'W2M_san': [W2M_san],
            'T2M_dav': [T2M_dav],
            'QV2M_dav': [QV2M_dav],
            'TQL_dav': [TQL_dav],
            'W2M_dav': [W2M_dav],
            'Holiday_ID': [0],
            'holiday': [1],
            'school': [0]
        }
        sample_input_df = pd.DataFrame(sample_input_data)
        #sample_input_df.index=sample_input_df['datetime']

        combined_df = pd.concat([df,sample_input_df],ignore_index=True)
        cat_type = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
        df=combined_df
        #df.index = pd.to_datetime(df.index)
        df['date'] = df.index
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekday'] = df['date'].dt.day_name()
        df['weekday'] = df['weekday'].astype(cat_type)
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.isocalendar().week  # Use isocalendar().week
        df['date_offset'] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300
        df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])
        df.sort_values(by=['datetime'],inplace=True)
        numerical_features = ['T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc', 'T2M_san', 'QV2M_san', 
                            'TQL_san', 'W2M_san', 'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav','Holiday_ID', 'holiday', 'school', 'hour','quarter', 
                                'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
        categorical_features = ['weekday' , 'season']
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_numerical = scaler.fit_transform(df[numerical_features])
        encoder = OneHotEncoder(sparse=False)
        X_categorical = encoder.fit_transform(df[categorical_features])
        X = np.concatenate([X_numerical, X_categorical], axis=1)

        # Prepare target variable
        y = df['nat_demand'].values
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        last_row_reshaped=X[-1].reshape(1, 1, -1)
        X_reshaped=X_reshaped[:-1]
        y=y[:-1]
        train_size = int(len(X_reshaped) * 0.8)
        test_size = len(X_reshaped) - train_size
        X_train, X_test = X_reshaped[0:train_size], X_reshaped[train_size:len(X_reshaped)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        #function for lstm
        lstm_model_path = "models\lstm_model.joblib"
        if not os.path.exists(lstm_model_path):
            lstm_model = TabularLSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
            # lstm_model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
            # lstm_model.add(Dense(1))
            lstm_model.compile()
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
            lstm_model.save_model(lstm_model_path)
        else:
            lstm_model = joblib.load(lstm_model_path)

        predicted_demand = lstm_model.predict(last_row_reshaped)
        st.text("Results using LSTM")
        st.dataframe(predicted_demand)


        #function for gru
        gru_units = [64, 64]  # Number of GRU units in each layer
        input_shape = (X_train.shape[1], X_train.shape[2])  # Input shape

        gru_model_path = "models\gru_model.joblib"
        if not os.path.exists(gru_model_path):
            gru_model = TabularGRUModel(input_shape=input_shape)
            gru_model.compile()
            gru_model.fit(X_train, y_train, epochs=50) 
            gru_model.save_model(gru_model_path)
        else:
            gru_model = joblib.load(gru_model_path)

        predictions_gbu = gru_model.predict(last_row_reshaped)
        st.text("Results using GRU")
        st.dataframe(predictions_gbu)

        # predictions={"LSTM":predicted_demand,"GRU":predictions_gbu}
        # df1_pred=pd.DataFrame(predictions)
        # df1_pred.index.name = "nat_demand"
        # st.dataframe(df1_pred)

        X_reshaped_2D = X_reshaped.squeeze(axis=1)
        last_row_reshaped_2D= last_row_reshaped.squeeze(axis=1)
        train_size1 = int(len(X_reshaped_2D) * 0.8)
        test_size1 = len(X_reshaped_2D) - train_size
        X_train1, X_test1 = X_reshaped_2D[0:train_size1], X_reshaped_2D[train_size1:len(X_reshaped_2D)]
        y_train1, y_test1 = y[0:train_size1], y[train_size1:len(y)]


        regression_model = RegressionModels()
        trained_models = regression_model.train_regression_models(X_train1, y_train1)
        predictions = regression_model.make_predictions(trained_models, last_row_reshaped_2D)
        st.text("Results using Regression Models")
        #st.dataframe(predictions)
        predictions_dict = {"Predictions": predictions}
        df_predictions = pd.DataFrame(predictions)
        df_predictions.index.name = "nat_demand"
        st.dataframe(df_predictions)

    

if __name__=="__main__": 
    main() 
