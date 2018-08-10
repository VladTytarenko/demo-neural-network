import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from math import fabs
from datetime import datetime
from clickhouse_driver.client import Client
from mqtt import MqttConnect

COLUMNS_DICT = {'0AA0/0003': 'luminosity', '0AA0/0006': 'humidity',  # '0AA0/0001': 'beside',
                '0AA0/0000': "inside", '0AA0/0002': 'outside'}
SECONDS = 60
INTERVAL = 15
PREDICTION_NUMBER = 12


def csv_data():
    """ """

    df = pd.read_csv("greenhouse.csv")
    df.set_index(keys='ts', inplace=True)

    df_x = df[df.columns[:5]]
    df_y = df[df.columns[5:]]

    head_n = 16800
    tail_n = 6200
    # for i in df_x.columns:
    #     plt.plot(df_x.head(head_n)[i], color='black')
    #     plt.title(i)
    #     plt.show()
    #
    # for i in df_x.columns:
    #     plt.plot(df_x.tail(tail_n)[i], color='blue')
    #     plt.title(i)
    #     plt.show()

    x_train = np.array(df_x.head(head_n)).astype(np.float)
    x_test = np.array(df_x.head(head_n + tail_n).tail(tail_n)).astype(np.float)
    y_train = np.array(df_y.head(head_n)).astype(np.float)
    y_test = np.array(df_y.head(head_n + tail_n).tail(tail_n)).astype(np.float)

    np.random.seed(42)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    # x_valid -= mean
    # x_valid /= std

    return x_train, y_train, x_test, y_test


def greenhouse():
    """ """

    conn = sqlite3.connect("/home/vladyslav/Downloads/greengarden.db")

    cursor = conn.cursor()

    cursor.execute("SELECT data_id, year, month, date, hour, minute, data FROM log")
    data = np.array(cursor.fetchall())
    conn.close()

    df = pd.DataFrame(data, columns=['topic', 'year', 'month', 'date', 'hour', 'minute', 'v'])
    df['ts'] = list(map(lambda x: int(datetime(x[0], x[1], x[2], x[3], x[4]).timestamp()),
                        list(zip(df['year'], df['month'], df['date'], df['hour'], df['minute']))))
    df.drop(columns=['year', 'month', 'date', 'hour', 'minute'], inplace=True)

    df['topic'] = list(map(str, df['topic']))
    df['topic'] = "0AA0/000" + df['topic']
    df['v'] /= 100

    return df


def get_data_set():
    """ Function for getting dataset from db. """

    conn = sqlite3.connect("/home/vladyslav/Desktop/greengarden.db")

    cursor = conn.cursor()

    cursor.execute("SELECT outside, inside, beside, humidity, luminosity FROM prediction LIMIT 4000")
    x_train = np.array(cursor.fetchall())

    # cursor.execute("SELECT t1, t2, t3, t4, t5, t6, t7 ,t8, t9, t10, t11, t12 FROM prediction LIMIT 5000")
    cursor.execute("SELECT t12 FROM prediction LIMIT 4000")
    y_train = np.array(cursor.fetchall())

    cursor.execute("SELECT outside, inside, beside, humidity, luminosity FROM prediction LIMIT 4000, 1000")
    x_test = np.array(cursor.fetchall())

    # cursor.execute("SELECT t1, t2, t3, t4, t5, t6, t7 ,t8, t9, t10, t11, t12 FROM prediction LIMIT 5000, 1000")
    cursor.execute("SELECT t12 FROM prediction LIMIT 4000, 1000")
    y_test = np.array(cursor.fetchall())
    conn.close()

    np.random.seed(42)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    return x_train, y_train, x_test, y_test


def clickhouse_dataset(predictions_number, interval, pred=None):
    """ Function for getting dataset from clickhouse database."""

    # client = Client(host="195.201.28.39")
    # data = np.array(
    #     client.execute(
    #         "SELECT topic, toUInt64(toStartOfMinute(dt)), v FROM greenhouse ORDER BY ts DESC LIMIT 23000, 10000"))
    # client.disconnect()
    #
    # pd.options.mode.chained_assignment = None
    #
    # df = pd.DataFrame(data, columns=["topic", "ts", "v"])
    # e = greenhouse()
    # df = df.append(e, sort=True)
    # df.set_index(keys="ts", inplace=True)
    #
    # t = df[df['topic'] == "0AA0/0001"]
    # t.drop(columns=["topic"], inplace=True)
    # t.rename(columns={'v': "beside"}, inplace=True)
    #
    # df_list = []
    #
    # for i in COLUMNS_DICT:
    #     temp = df[df['topic'] == i]
    #     temp.drop(columns=["topic"], inplace=True)
    #     temp.rename(columns={'v': COLUMNS_DICT[i]}, inplace=True)
    #     df_list.append(temp)
    # res = t.join(df_list, how='inner')
    # df_list.clear()
    # res.rename(index=lambda x: int(x), inplace=True)

    # TODO make data normalize

    res = pd.read_csv("greenhouse.csv")[['ts', 'outside', 'inside', 'beside', 'humidity', 'luminosity']]
    res.rename(index=lambda x: int(x), inplace=True)
    res['ts'] = res['ts'].apply(lambda x: int(x))
    df_list, predictions_columns_list = list(), list()

    for i in range(1, predictions_number + 1):
        temp = res['beside'].to_frame()
        columns_name = 't' + str(i)
        predictions_columns_list.append(columns_name)
        temp.rename(index=lambda x: x - (i * interval * SECONDS),
                    columns={'beside': columns_name},
                    inplace=True)
        df_list.append(temp)

    res = res.join(df_list, how='inner')

    res.set_index('ts', inplace=True)

    day = 86400
    delta = int(day * 0.6)
    s = 1512570481
    for i in range(48):
        drop_index = res.loc[s:s+delta].index
        if not drop_index.empty:
            res.drop(drop_index, inplace=True)
        s += day

    df_x = res[['outside', 'inside', 'beside', 'humidity', 'luminosity']]
    if pred is None:
        df_y = res[predictions_columns_list]
    else:
        df_y = res[[pred]]

    head_n, tail_n = 15000, 6200
    # 1512543180
    x_train = np.array(df_x.head(head_n)).astype(np.float)
    x_test = np.array(df_x.head(head_n + tail_n).tail(tail_n)).astype(np.float)
    y_train = np.array(df_y.head(head_n)).astype(np.float)
    y_test = np.array(df_y.head(head_n + tail_n).tail(tail_n)).astype(np.float)

    np.random.seed(42)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    return x_train, y_train, x_test, y_test


def test(pred, y_test):
    """ Function test make compare of prediction with real temperature. """

    max_delta, n_max, min_delta, n_min, delta_out, mean = 0, 0, 0, 0, 0, 0
    file = open("result.txt", 'w')
    file.write("Прогнозована температура | Pеальна температура\n")
    for i in range(len(pred)):
        file.write(str(i) + '\n')
        mean_block = 0
        delta_list = []
        for p, t in zip(pred[i], y_test[i]):
            delta = t - p
            delta_list.append(delta)
            mean += fabs(delta)
            f = str('%.5f' % p) + ' | ' + str('%.5f' % t) + " delta = " + str('%.5f' % delta) + '\n'
            file.write(f)
            mean_block += delta
            if max_delta < delta:
                (max_delta, n_max) = delta, i
            if min_delta > delta:
                (min_delta, n_min) = delta, i
            if delta >= 3 or delta <= -3:
                delta_out += 1
        mean_block /= len(pred[i])
        mbt = "ME" + str(mean_block) + '\n'
        file.write(mbt)
        file.write("**********************\n")
        # if i % 3 == 0:
        #     plt.plot(y_test[i], color='black', label='Original data')
        #     plt.plot(pred[i], color='blue', label='Predicted data')
        #     plt.legend(loc='best')
        #     plt.title('Actual and predicted')
        #     plt.show()
        # plt.plot(delta_list, color='red', label='delta')
        # plt.ylim(-3, 3)
        # plt.legend(loc='best')
        # me = 'mean error {0}'.format(str(mean_block))
        # plt.title(me)
        # plt.show()

    mean /= len(pred) * len(pred[0])
    log = "Середня помилка " + str(mean) + "\nМаксимальна похибка(+) " + str(
        max_delta) + " номер тестового прикладу " + str(n_max) + "\nМаксимальна похибка(-) " + str(
        min_delta) + " номер тестового прикладу " + str(n_min) + "\nmax-min " + str(
        max_delta - min_delta) + "\nКількість прогнозів з помилкою більшою за допустиму " + str(delta_out)
    file.write(log)
    print(log)

    # mqtt = MqttConnect()
    # mqtt.publish_json(pred[64])


#
# client = Client(host="195.201.28.39")
# data = np.array(
#     client.execute(
#         "SELECT * FROM greengarden_prediction_n"))
# client.disconnect()
# df = pd.DataFrame(data, columns=['ts', 'outside', 'inside', 'beside', 'humidity', 'luminosity',
#                                  't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10',
#                                  't11', 't12'])
#
# temp = df[df['outside'] > 100]
# df.set_index(keys=['ts'], inplace=True)
# df.drop(index=temp['ts'], inplace=True)
# col = ['inside', 'beside', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8',
#        't9', 't10', 't11', 't12']
# for i in col:
#     for j in df[df[i] > 100].index:
#         df.at[j, i] /= 10
#
# # for i in col:
# #     if df[df[i] > 100].empty:
# #         print("ok")
# # colors = ['black', 'red',
# #           'blue', 'green', 'yellow']
# # for i, j in zip(df.columns[0:5], colors):
# #     plt.plot(df[i], color=j, label=i)
# # plt.legend(loc='best')
# # plt.show()
#
# df.to_csv('greenhouse.csv')

# x_train, y_train, x_test, y_test, x_valid = clickhouse_dataset(12, 15)
# df = pd.DataFrame(x_test, columns=list(map(str, range(12))))
# for i in y_test:
#     plt.plot(i, color='black')
#     plt.show()
