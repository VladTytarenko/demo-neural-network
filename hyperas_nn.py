from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ELU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Nadam, Adamax

from hyperas import optim
from hyperas.distributions import choice
from data_set import csv_data


def data():
    x_train, y_train, x_test, y_test = csv_data()
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
# glorot_u, he_u, adam
    model = Sequential(name="first")
    model.add(Dense(60, kernel_initializer={{choice(["he_uniform", "glorot_uniform"])}}, input_shape=(x_train.shape[1],)))
    model.add(ELU())
    model.add(BatchNormalization())

    for i in range(3):  # 3
        model.add(Dense(168, kernel_initializer={{choice(["he_uniform", "glorot_uniform"])}}))
        model.add(ELU())
        model.add(BatchNormalization())

    model.add(Dense(12))

    callbacks_list = [
        ReduceLROnPlateau(monitor="mean_absolute_error",
                          factor=0.8,
                          patience=5,
                          min_lr=0.000001,
                          verbose=1),
        EarlyStopping(monitor="mean_absolute_error",
                      patience=22,
                      verbose=1),
        ModelCheckpoint(filepath='neural_network.h5',
                        monitor='mean_absolute_error',
                        save_best_only=True,
                        verbose=1)
    ]

    model.compile(optimizer={{choice([Adam(), Nadam(), Adamax()])}}, loss="mse", metrics=["mae"])
    model.fit(x_train, y_train,
              epochs=200,
              batch_size=16,
              verbose=2,
              callbacks=callbacks_list)
    mse, mae = model.evaluate(x_test, y_test, verbose=0)
    print('MAE:', mae)
    return {'loss': mse, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
