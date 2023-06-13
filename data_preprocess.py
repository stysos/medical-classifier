import numpy as np
import pandas as pd
import arff
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(open_object):
    """

    :param open_object:
    :return: dataframe
    """
    data_arff = arff.loads(open_object)

    colnames = []
    for i in range(len(data_arff['attributes'])):
      colnames.append(data_arff['attributes'][i][0])

    dataframe = pd.DataFrame.from_dict(data_arff['data'])
    dataframe.columns = colnames
    print(f'\nLoading\nData loaded of length {len(dataframe)}!\n{dataframe.columns}')
    return dataframe



def drop_columns(dataframe, *columns):
    """
    @:param dataframe
    @:returns dataframe, dataframe_additional

    """
    dataframe_additional = dataframe[list(columns)]
    print(f'\nCreated additional dataframe from dropped')
    dataframe = dataframe.drop(list(columns), axis=1)
    print(f'\nPreprocessing:\nDropped {columns} columns')
    return dataframe, dataframe_additional


def binary_encoding(dataframe, *columns):
    """

    :param dataframe: takes and stores from the dataframe
    :param columns: encodes these columns (yes or no only)
    :return: the dataframe with binary encoded columns
    """
    print(f'\nEncoding:\nRunning encoding on {columns}')
    for col in columns:
        encoded = []
        for entry in dataframe[col]:
            if entry.lower() == 'yes':
                encoded.append(1)
            elif entry.lower() == 'no':
                encoded.append(0)
            elif entry.lower() == 'm':
                encoded.append(1)
            elif entry.lower() == 'f':
                encoded.append(0)
        if col == 'gender':
            dataframe['male'] = encoded
        else:
            dataframe['has_' + col] = encoded
        print(f'Encoded {col}')
        dataframe = dataframe.drop(col, axis=1)
        print(f'Dropped {col}')
    return dataframe

def remove_categorical(dataframe, dataframe_additional, *columns):
    """
    removes categorical for further thinking and stores onto additional dataframe
    :param dataframe:
    :param columns:
    :return:
    """
    for col in columns:
        dataframe_additional[col] = dataframe[col]
        print(f'{col} added to additional')
        dataframe.drop(col, axis=1, inplace=True)
        print(f'{col} dropped from dataframe')
    return dataframe, dataframe_additional



def locate_na_index(dataframe):
    """
    Checks max of column containing na and return max slice to check for anomaly
    then replace na with mean after removing max
    :param dataframe:
    :return: na replaced with column.mean() dataframe
    """
    for indx, sum in enumerate(dataframe.isna().sum()):
        if sum > 0:
            print(f'index {indx} has {sum} null values')
            print(f'max slice:\n{dataframe.loc[dataframe.iloc[:, indx] == dataframe[dataframe.columns[indx]].max()]}')
            dataframe = dataframe.drop(
            dataframe.loc[dataframe.iloc[:, indx] == dataframe[dataframe.columns[indx]].max()].index
            )

            null_column = dataframe.iloc[:, indx]
            print(f'Typo case dropped {null_column.max()=} \n{null_column.mean()} mean')

            print(dataframe[dataframe[dataframe.columns[indx]].isnull()])
            for null_index in dataframe[dataframe[dataframe.columns[indx]].isnull()].index:
                dataframe = dataframe.drop(null_index)
                print(f'Dropped null case at {null_index=}')

            return dataframe


def correct_types(dataframe):
    """

    :param dataframe: dataframe to correct types of
    :return: dataframe with all column.values as type(int)
    """
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].astype(int)
        print(f'{col} is now type {type(dataframe[col][0])}')
    return dataframe


def create_training_variables(dataframe):
    """
    :param dataframe: dataframe to create training variables from
    :return: X -> pd.DataFrame(), x -> np.array, y -> np.array
    """
    X = dataframe.drop('has_ASD', axis=1)
    array_labels = X.columns
    y = np.array(dataframe['has_ASD'])
    x = np.array(dataframe.drop(['has_ASD'], axis=1))
    return X, x, y

def split_train_test(x, y):
    """

    :param x: np.array() features
    :param y: np.array() classification
    :return: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

def create_scaler(x_train, x_test):
    """

    :param x_train: scaler.fit_transform(x_train))
    :param x_test: scaler.transform(x_test)
    :return: x_train, x_test, scaler
    """
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, scaler

def create_model(x_train, y_train):
    """

    :param x_train: required for fit
    :param y_train: required for fit
    :return: regressor - trained
    """
    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(x_train, y_train)
    return regressor

def test_model(model, x_test, y_test):
    """

    :param model: regressor model to predict with
    :param x_test: values to predict
    :param y_test: correct y-values
    :return: y_predicted, score
    """
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    return y_pred, score


if __name__ == '__main__':
    path_arff = open('data/Autism-Adult-Data.arff')
    df = load_data(path_arff)
    df, df_additional = drop_columns(df, 'used_app_before', 'country_of_res', 'age_desc')
    df = binary_encoding(df, 'jundice', 'autism_relation', 'ASD', 'gender')
    df, df_additional = remove_categorical(df, df_additional, 'ethnicity', 'relation')
    df = locate_na_index(df)
    df = correct_types(df)
    X, x, y = create_training_variables(df)
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    # x_train, x_test, scaler = create_scaler(x_train, x_test) scaler removed as not needed (mostly binary values)
    regressor = create_model(x_train, y_train)

    y_pred, score = test_model(regressor, x_test, y_test)