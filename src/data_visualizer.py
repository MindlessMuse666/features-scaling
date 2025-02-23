import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class DataVisualizer:
    '''
    Класс для визуализации данных.
    '''

    def __init__(self):
        '''
        Инициализирует объект DataVisualizer.
        '''
        pass


    def create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str, title: str, window_title: str):
        '''
        Создает диаграмму рассеяния.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            x_column (str): Название столбца для оси X.
            y_column (str): Название столбца для оси Y.
            title (str): Заголовок графика.
            window_title (str): Заголовок окна графика.
        '''
        plt.figure(window_title)
        sns.scatterplot(x=x_column, y=y_column, data=df)
        plt.title(title)
        plt.show()


    def create_scatter_plot_plotly(self, df: pd.DataFrame, x_column: str, y_column: str, color_column: str, title: str):
        '''
        Создает интерактивную диаграмму рассеяния с помощью Plotly.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            x_column (str): Название столбца для оси X.
            y_column (str): Название столбца для оси Y.
            color_column (str): Название столбца для разделения по цвету.
            title (str): Заголовок графика.
        '''
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
        fig.show()


    def create_histograms(self, df: pd.DataFrame, title: str, window_title: str):
        '''
        Создает гистограммы для всех числовых признаков в DataFrame с использованием subplots и Seaborn.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            title (str): Заголовок для всего набора гистограмм.
            window_title (str): Заголовок окна с гистограммами.
        '''
        num_cols = len(df.columns)
        fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(8, 2.5 * num_cols))
        fig.canvas.manager.set_window_title(window_title)

        for i, column in enumerate(df.columns):
            ax = axes[i]
            sns.histplot(df[column], ax=ax)
            ax.set_title(column)
            ax.set_xlabel('')

        plt.suptitle(title, fontsize=13.5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0, w_pad=0.5, h_pad=1.0)
        plt.show()