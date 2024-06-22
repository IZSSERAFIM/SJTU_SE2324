import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['paInput'].flatten(), data['paOutput'].flatten()

def preprocess_data(pa_input, pa_output, sequence_length=10):
    X = []
    Y = []

    for i in range(len(pa_input) - sequence_length):
        X.append(np.hstack((np.real(pa_input[i:i + sequence_length]), np.imag(pa_input[i:i + sequence_length]))))
        Y.append(np.hstack((np.real(pa_output[i:i + sequence_length]), np.imag(pa_output[i:i + sequence_length]))))

    return np.array(X), np.array(Y)

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_shape, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_model(Y_pred, Y_test):
    Iout_pred = Y_pred[:, :Y_pred.shape[1]//2]
    Iout_test = Y_test[:, :Y_test.shape[1]//2]
    Qout_pred = Y_pred[:, Y_pred.shape[1]//2:]
    Qout_test = Y_test[:, Y_test.shape[1]//2:]

    nmse = 10 * np.log10(
        np.sum((Iout_test - Iout_pred)**2 + (Qout_test - Qout_pred)**2) /
        np.sum(Iout_test**2 + Qout_test**2)
    )

    return nmse

def main():
    pa_input_train, pa_output_train = load_data('../dataset/task1/PA_data_train.mat')
    pa_input_test, pa_output_test = load_data('../dataset/task1/PA_data_test.mat')

    X_train, Y_train = preprocess_data(pa_input_train, pa_output_train)
    X_test, Y_test = preprocess_data(pa_input_test, pa_output_test)

    # Print shapes for debugging
    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}')

    input_shape = X_train.shape[1]
    
    model = build_model(input_shape)
    model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)

    Y_pred = model.predict(X_test)

    nmse = evaluate_model(Y_pred, Y_test)

    print('Task 1')
    print(f'Normalized Mean Square Error (NMSE): {nmse:.4f}')

if __name__ == "__main__":
    main()
