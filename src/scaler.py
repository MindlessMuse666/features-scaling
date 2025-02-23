import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class Scaler:
    '''
    Класс для масштабирования признаков.
    '''

    def __init__(self):
        '''
        Инициализирует объект Scaler.
        '''
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()


    def scale_standard(self, df: pd.DataFrame, columns: list):
        '''
        Масштабирует данные с использованием StandardScaler.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            columns (list): Список столбцов для масштабирования.

        Returns:
            pandas.DataFrame: DataFrame с масштабированными данными.
        '''
        scaled_data = self.standard_scaler.fit_transform(df[columns])
        return pd.DataFrame(scaled_data, columns=columns)


    def scale_min_max(self, df: pd.DataFrame, columns: list):
        '''
        Масштабирует данные с использованием MinMaxScaler.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            columns (list): Список столбцов для масштабирования.

        Returns:
            pandas.DataFrame: DataFrame с масштабированными данными.
        '''
        scaled_data = self.min_max_scaler.fit_transform(df[columns])
        return pd.DataFrame(scaled_data, columns=columns)


    def scale_robust(self, df: pd.DataFrame, columns: list):
        '''
        Масштабирует данные с использованием RobustScaler.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            columns (list): Список столбцов для масштабирования.

        Returns:
            pandas.DataFrame: DataFrame с масштабированными данными.
        '''
        scaled_data = self.robust_scaler.fit_transform(df[columns])
        return pd.DataFrame(scaled_data, columns=columns)