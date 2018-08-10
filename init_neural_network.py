import data_set
from data_set import clickhouse_dataset
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ELU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


def base_network(input_parameters, x_train, y_train):
    """Base model for single temperature prediction.
    #Arguments:
        input_parameters: number of input parameter for prediction temperature
    #Returns:
        trained neural network
    """

    model = Sequential()
    model.add(Dense(60, kernel_initializer="he_uniform", input_shape=input_parameters))
    model.add(ELU())
    model.add(BatchNormalization())

    for i in range(3):
        model.add(Dense(168, kernel_initializer="he_uniform"))
        model.add(ELU())
        model.add(BatchNormalization())

    model.add(Dense(1, kernel_initializer="glorot_uniform"))
    model.add(BatchNormalization())

    callbacks_list = [
        ReduceLROnPlateau(monitor="mean_absolute_error",
                          factor=0.8,
                          patience=5,
                          min_lr=0.000001,
                          verbose=1),
        EarlyStopping(monitor="mean_absolute_error",
                      patience=20,
                      verbose=1),
        ModelCheckpoint(filepath='neural_network.h5',
                        monitor='mean_absolute_error',
                        save_best_only=True,
                        verbose=1)
    ]

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=2, callbacks=callbacks_list)
    return model


def final_prediction():
    num = 12
    interval = 15
    x_train, y_train, x_test, y_test = clickhouse_dataset(predictions_number=num, interval=interval)
    model_list = list()
    for st in y_train:
        model_list.append(base_network((x_train.shape[1], ), x_train, st))

    for i in range(num):
        data_set.test(model_list[i].predict(x_test), y_test[i])


final_prediction()