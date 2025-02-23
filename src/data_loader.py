import pandas as pd


class DataLoader:
    '''
    Класс для загрузки данных из CSV-файла.
    '''

    def __init__(self, file_path):
        '''
        Инициализирует объект DataLoader.

        Args:
            file_path (str): Путь к CSV-файлу.
        '''
        self.file_path = file_path


    def load_data(self):
        '''
        Загружает данные из CSV-файла.

        Returns:
            pandas.DataFrame: DataFrame с данными.
        '''
        try:
            df = pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            print(f'Ошибка: Файл не найден по пути {self.file_path}')
            return None
        except Exception as e:
            print(f'Ошибка при загрузке данных: {e}')
            return None
