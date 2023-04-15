import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

rejected_labels = [
    'Unnamed: 0.1',
    'Unnamed: 0',
    'version',
    'alive',
    'tagId',
    'success',
    'timestamp',
    'data__tagData__pressure',
    'data__anchorData',
    'data__coordinates__x',
    'data__coordinates__y',
    'data__coordinates__z',
    'reference__x',
    'reference__y',
    'errorCode'
]


def load_xlsx_file(filename):
    input_data = pd.read_excel(filename)
    output_data = DataFrame()
    output_data['reference__x'] = input_data['reference__x']
    output_data['reference__y'] = input_data['reference__y']

    for index, row in input_data.iterrows():
        if row['success'] is False:
            input_data = input_data.drop(index)
            output_data = output_data.drop(index)

    for column_name in rejected_labels:
        if column_name in input_data:
            input_data.pop(column_name)

    return input_data, DataFrame(output_data)


def load_training_data(room_number):
    training_input_data = DataFrame()
    training_output_data = DataFrame()
    if room_number == 8:
        for index in range(1, 226):
            data = load_xlsx_file('data/pomiary/F8/f8_stat_' + str(index) + '.xlsx')
            training_input_data = pd.concat([training_input_data, data[0]])
            training_output_data = pd.concat([training_output_data, data[1]])
    if room_number == 10:
        for index in range(1, 226):
            data = load_xlsx_file('data/pomiary/F10/f10_stat_' + str(index) + '.xlsx')
            training_input_data = pd.concat([training_input_data, data[0]])
            training_output_data = pd.concat([training_output_data, data[1]])
    return training_input_data, training_output_data


def load_testing_data(room_number):
    testing_input_data = DataFrame()
    testing_output_data = DataFrame()
    if room_number == 8:
        for index in range(1, 4):
            data = load_xlsx_file('data/pomiary/F8/f8_' + str(index) + 'p.xlsx')
            testing_input_data = pd.concat([testing_input_data, data[0]])
            testing_output_data = pd.concat([testing_output_data, data[1]])
    if room_number == 10:
        for index in range(1, 4):
            data = load_xlsx_file('data/pomiary/F10/f10_' + str(index) + 'p.xlsx')
            testing_input_data = pd.concat([testing_input_data, data[0]])
            testing_output_data = pd.concat([testing_output_data, data[1]])
    return testing_input_data, testing_output_data


def plot(data):
    plt.bar(np.arange(0, len(data[0])), data[0], color = 'green' )
    plt.title('Pierwiastek błędu średniokwadratowego dla zbioru testowego')
    plt.xlabel('Numer pomiaru')
    plt.ylabel('RMSE')
    plt.show()

    plt.bar(np.arange(0, len(data[1])), data[1], color = 'green')
    plt.title('Średnia arytmetyczna z wag danych wejściowych')
    plt.xlabel('Indeks kolumny wejściowej')
    plt.ylabel('Waga kolumny')
    plt.show()


def solve(training_input, training_output, testing_input, testing_output):
    training_input_tensor = tf.convert_to_tensor((training_input.astype('float32')) / 10000)
    training_output_tensor = tf.convert_to_tensor((training_output.astype('float32')) / 10000)
    testing_input_tensor = tf.convert_to_tensor((testing_input.astype('float32')) / 10000)
    testing_output_tensor = tf.convert_to_tensor((testing_output.astype('float32')) / 10000)

    model = tf.keras.Sequential([                                               #liczba epok - 25
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(23,)),    #Warstwa wejściowa - 23 neurony,I warstwa ukryta - 64 neurony, funkcja aktywacyjna to ReLU
        tf.keras.layers.Dense(32, activation=tf.nn.relu),                       #II warstwa ukryta - 32 neurony, funkcja aktywacyjna to ReLU
        tf.keras.layers.Dense(16, activation=tf.nn.relu),                       #III warstwa ukryta - 16 neuronów, funkcja aktywacyjna to ReLU
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),                     #Werstwa wyjściowa - 2 neurony, funkcja aktywacyjna to funkcja sigmoidalna
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.MeanSquaredError(), metrics=['accuracy'])

    model.fit(training_input_tensor, training_output_tensor, epochs=25)
    model.evaluate(testing_input_tensor, testing_output_tensor)
    weights = model.layers[0].get_weights()[0]
    pr = model.predict(testing_input_tensor)
    result = (pr * 10000)
    result_df = DataFrame(result)

    mse = []
    for index in range(0, len(result_df[0])):
        mse_xy = mean_squared_error([result_df[0][index], result_df[1][index]],
                                    [DataFrame(testing_output['reference__x']).iloc[0],
                                     DataFrame(testing_output['reference__y']).iloc[0]])
        mse.append(mse_xy)

    mse = np.sqrt(mse)
    print(mse)
    print(np.mean(weights, axis=1))
    plot([mse, np.mean(weights, axis=1)])
    return mse


def save_to_xlsx(filename, data):
    DataFrame(data).to_excel(filename)


def main():
    # Room F8
    training_input, training_output = load_training_data(8)
    testing_input, testing_output = load_testing_data(8)

    mse8 = solve(training_input, training_output, testing_input, testing_output)

    # Room F10
    training_input, training_output = load_training_data(10)
    testing_input, testing_output = load_testing_data(10)

    mse10 = solve(training_input, training_output, testing_input, testing_output)

    mse = np.append(mse8, mse10)

    save_to_xlsx("wyniki.xlsx", mse)


main()
