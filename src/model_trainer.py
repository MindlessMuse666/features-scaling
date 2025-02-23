import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ModelTrainer:
    '''
    Класс для обучения и оценки модели машинного обучения.
    '''

    def __init__(self, test_size=.2, random_state=42):
        '''
        Инициализирует объект ModelTrainer.

        Args:
            test_size (float): Размер тестового набора.
            random_state (int): Зерно для случайного разделения.
        '''
        self.test_size = test_size
        self.random_state = random_state
        self.model = LogisticRegression(solver='liblinear', random_state=self.random_state)


    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Обучает модель логистической регрессии.

        Args:
            X_train (pd.DataFrame): Обучающие признаки.
            y_train (pd.Series): Обучающие метки.
        '''
        self.model.fit(X_train, y_train)


    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        '''
        Оценивает производительность модели на тестовом наборе.

        Args:
            X_test (pd.DataFrame): Тестовые признаки.
            y_test (pd.Series): Тестовые метки.

        Returns:
            float: Точность модели на тестовом наборе.
        '''
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    def split_data(self, X: pd.DataFrame, y: pd.Series):
        '''
        Разделяет данные на обучающий и тестовый наборы.

        Args:
            X (pd.DataFrame): Признаки.
            y (pd.Series): Метки.

        Returns:
            tuple: Кортеж из X_train, X_test, y_train, y_test.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test