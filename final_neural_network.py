import keras
import data_set
from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from keras.utils.vis_utils import plot_model


NUMBER_OF_HIDDEN_LAYERS = 5
EPOCHS = 100
BATCH_SIZE = 12


def init_neural_network(prediction_number, interval):
    """Creating and training ensembling neural network for temperature
    prediction. Function create NN in ensemble for each single prediction.
    Testing NN on testing dataset and print statistics.

    #Arguments:
        prediction_number: total predictions' number of result. The number
        of NN in the ensemble is the same as the value of the argument.

    #Returns:
        trained neural network
    """

    # x_train, y_train, x_test, y_test = data_set.get_data_set()
    x_train, y_train, x_test, y_test = data_set.clickhouse_dataset(predictions_number=prediction_number,
                                                                   interval=interval)

    nn_list = []
    out_list = []

    for i in range(prediction_number):

        x = Input(shape=(x_train.shape[1],))
        x = Dense(5, kernel_initializer="glorot_normal")(x)  # activation="sigmoid"
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        for j in range(NUMBER_OF_HIDDEN_LAYERS):
            x = Dense(14,  # activation="sigmoid",
                      kernel_initializer="he_normal")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
        input_layer = Dense(1, kernel_initializer="he_normal")(x)
        out_list.append(x)
        nn_list.append(input_layer)

    concatenate = keras.layers.concatenate(inputs=nn_list)
    # out = Dense(prediction_number)(concatenate)
    model = Model(inputs=nn_list, outputs=concatenate, name="main")
    # model = Model(inputs=nn_list, outputs=out_list)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    print(model.summary())
    plot_model(model, show_shapes=True)
    print(type(x_train))
    print(x_train)
    model.fit([x_train] * prediction_number, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    mse, mae = model.evaluate([x_test] * prediction_number, y_test, verbose=0)
    print("Средня абсолютна помилка (градусів цельсія):", mae)
    print(mse)

    pred = model.predict([x_test] * prediction_number)

    data_set.test(pred, y_test)


init_neural_network(12, 15)