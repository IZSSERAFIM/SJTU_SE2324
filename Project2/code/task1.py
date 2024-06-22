import scipy.io
import numpy as np

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['paInput'], data['paOutput']

def preprocess_data(pa_input, pa_output):
    X_real = np.real(pa_input)
    X_imag = np.imag(pa_input)
    Y_real = np.real(pa_output)
    Y_imag = np.imag(pa_output)

    X = np.hstack((X_real, X_imag))
    Y = np.hstack((Y_real, Y_imag))

    return X, Y

def simple_fit(X_train, Y_train, X_test):
    Y_pred = np.zeros(X_test.shape)
    for i in range(X_test.shape[1]):
        Y_pred[:, i] = np.interp(X_test[:, i], X_train[:, i], Y_train[:, i])
    return Y_pred

def evaluate_model(Y_pred, Y_test):
    Iout_pred = Y_pred[:, Y_pred.shape[1]//2:]
    Iout_test = Y_test[:, Y_test.shape[1]//2:]
    Qout_pred = Y_pred[:, Y_pred.shape[1]//2:]
    Qout_test = Y_test[:, Y_test.shape[1]//2:]

    nmse = 10 * np.log10(
        np.sum((Iout_test - Iout_pred)**2 + (Qout_test - Qout_pred)**2) /
        np.sum(Iout_test**2 + Qout_test**2)
    )

    return nmse

def main():
    pa_input_train, pa_output_train = load_data('../dataset/task1/PA_data_train.mat')
    X_train, Y_train = preprocess_data(pa_input_train, pa_output_train)

    pa_input_test, pa_output_test = load_data('../dataset/task1/PA_data_test.mat')
    X_test, Y_test = preprocess_data(pa_input_test, pa_output_test)

    Y_pred = simple_fit(X_train, Y_train, X_test)

    nmse = evaluate_model(Y_pred, Y_test)

    print(f'Normalized Mean Square Error (NMSE): {nmse:.4f}')

if __name__ == "__main__":
    main()
