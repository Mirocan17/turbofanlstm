# turbofanlstm
Remaining Useful Life Analysis
# Turbofan Engine RUL (Remaining Useful Life) Prediction — LSTM

An LSTM model that predicts the Remaining Useful Life (RUL) of turbofan engines
from multivariate sensor readings, trained on NASA's C-MAPSS (FD001) degradation
dataset.

## Dataset

- **Source:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001 subset)
- 100 engines, each run to failure over a different number of operating cycles
- 21 sensor measurements + 3 operational setting columns per cycle
- Each engine is monitored from a healthy state until failure (run-to-failure data)

## Method

1. **Preprocessing**
   - `RUL = maxCycle - timeCycle` computed for every row
   - **RUL capping** applied (upper bound: 125 cycles). Early in an engine's life,
     sensor readings look essentially identical regardless of whether the true RUL
     is 300 or 200 cycles — there's no learnable signal in that region. Capping
     lets the model focus on the degradation window where the data actually carries
     information, instead of being penalized for an unlearnable target.
   - Sensor features scaled with MinMaxScaler (fit on the training set only, to avoid
     leakage)
   - Sequences built with a sliding window over each engine's time series

2. **Data split**
   - Split at the **engine (unit) level** into train / validation / test, so that
     cycles from the same engine never appear in more than one split

3. **Model**
   - Single LSTM layer + Dense output layer
   - L2 regularization
   - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`

## Results

| Stage | J_train (MSE) | J_cv (MSE) | Ratio |
|---|---|---|---|
| Before RUL capping | 135.60 | 1193.23 | 8.8x |
| After RUL capping | 31.47 | 154.27 | 4.9x |
| + Early stopping | 29.08 | 137.47 | 4.7x |

**Test set performance:** MAE = 9.56 cycles, RMSE = 12.73 cycles

Per-engine error analysis showed 14 of 15 test engines predicted consistently and
accurately, while one engine (unit 67, RMSE = 20.83) was predicted with a
systematic negative bias — most likely due to unit-to-unit manufacturing/baseline
variation rather than a modeling flaw.

## Key Takeaways

- Training on uncapped RUL forces the model to chase an unlearnable target during
  an engine's healthy phase, inflating both training and validation error
- With a small validation set (10 engines here), a single hard-to-predict engine
  can noticeably skew the aggregate metric — per-unit error breakdown is essential
  for telling a real generalization problem apart from a single outlier

## Stack

- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib
