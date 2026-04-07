import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

import statsmodels.api as sm

# -------------------------
# DATASET (NO API)
# -------------------------
class Dataset:
    def load(self):
        years = np.arange(1970, 2020)

        df = pd.DataFrame({
            "year": years,
            "population": np.linspace(550, 1350, len(years)),
            "life_expectancy": np.linspace(50, 70, len(years)) + np.random.normal(0, 0.5, len(years)),
            "health_exp_gdp": np.linspace(2, 6, len(years)),
            "physicians": np.linspace(0.2, 0.9, len(years)),
            "hospital_beds": np.linspace(0.5, 1.5, len(years)),
            "infant_mortality": np.linspace(120, 30, len(years))
        })

        X = df.drop(columns=["year", "life_expectancy"]).values
        y = df["life_expectancy"].values

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        return df, X, y

    def create_sequences(self, X, y, window=8):
        Xs, ys = [], []
        for i in range(len(X) - window):
            Xs.append(X[i:i+window])
            ys.append(y[i+window])
        return np.array(Xs), np.array(ys)

# -------------------------
# MODELS
# -------------------------
class LSTMModel:
    def build(self, input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(4)(inputs)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

class TransformerModel:
    def build(self, input_shape):
        inputs = Input(shape=input_shape)

        attn = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
        x = LayerNormalization()(inputs + attn)

        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)

        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

# -------------------------
# CAUSAL INFERENCE
# -------------------------
class CausalInference:
    def run(self, df, policy_year=1995):
        df = df.copy()
        df['post'] = (df['year'] >= policy_year).astype(int)
        df['time'] = df['year'] - df['year'].min()
        df['interaction'] = df['post'] * df['time']

        X = df[['post', 'time', 'interaction']]
        X = sm.add_constant(X)
        y = df['life_expectancy']

        model = sm.OLS(y, X).fit()
        return model.summary()

# -------------------------
# SIMULATION
# -------------------------
class Simulation:
    def __init__(self):
        self.df, X, y = Dataset().load()
        self.X, self.y = Dataset().create_sequences(X, y)

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )

        es = EarlyStopping(patience=5, restore_best_weights=True)

        # LSTM
        lstm = LSTMModel().build((X_train.shape[1], X_train.shape[2]))
        lstm.fit(X_train, y_train, epochs=30, verbose=0,
                 validation_data=(X_test, y_test), callbacks=[es])
        pred_lstm = lstm.predict(X_test).flatten()

        # Transformer
        tr = TransformerModel().build((X_train.shape[1], X_train.shape[2]))
        tr.fit(X_train, y_train, epochs=30, verbose=0,
               validation_data=(X_test, y_test), callbacks=[es])
        pred_tr = tr.predict(X_test).flatten()

        # Linear baseline
        lr = LinearRegression()
        lr.fit(X_train.reshape(len(X_train), -1), y_train)
        pred_lr = lr.predict(X_test.reshape(len(X_test), -1))

        # Metrics
        print("LSTM:", mean_squared_error(y_test, pred_lstm), r2_score(y_test, pred_lstm))
        print("Transformer:", mean_squared_error(y_test, pred_tr), r2_score(y_test, pred_tr))
        print("Linear:", mean_squared_error(y_test, pred_lr), r2_score(y_test, pred_lr))

        # Causal
        print(CausalInference().run(self.df))

        # Plot
        plt.figure()
        plt.plot(y_test, label="Actual")
        plt.plot(pred_lstm, label="LSTM")
        plt.plot(pred_tr, label="Transformer")
        plt.plot(pred_lr, label="Linear")
        plt.legend()
        plt.savefig("final_plot.png")
        print("Saved: final_plot.png")

if __name__ == "__main__":
    Simulation().run()
