import data_set
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, ELU, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


x_train, y_train, x_test, y_test = data_set.clickhouse_dataset(predictions_number=12,
                                                               interval=20)

model = Sequential(name="first")
model.add(Dense(9, kernel_initializer="he_uniform", input_shape=(x_train.shape[1],)))
model.add(ELU())
model.add(BatchNormalization())

for i in range(2):
    model.add(Dense(9, kernel_initializer="he_uniform"))
    model.add(ELU())
    model.add(BatchNormalization())

model.add(Dense(12))

callbacks_list = [
    # ReduceLROnPlateau(monitor="mean_absolute_error",
    #                   factor=0.8,
    #                   patience=5,
    #                   min_lr=0.000001,
    #                  verbose=1),
    EarlyStopping(monitor="mean_absolute_error",
                  patience=22,
                  verbose=1),
    ModelCheckpoint(filepath='neural_network.h5',
                    monitor='mean_absolute_error',
                    save_best_only=True,
                    verbose=1)
]

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(x_train, y_train, epochs=150, batch_size=16, verbose=2, callbacks=callbacks_list)

mse, mae = model.evaluate(x_test, y_test, verbose=0)
pred = model.predict(x_test)
data_set.test(pred, y_test)
