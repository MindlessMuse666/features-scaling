from data_loader import DataLoader
from data_visualizer import DataVisualizer
from scaler import Scaler
from model_trainer import ModelTrainer


DATASET_URL = 'https://github.com/fcschmidt/knn_iris_dataset/blob/master/datasets/iris/Iris.csv?raw=true'

TEST_SIZE    = .2  # Константа размера тестового набора
RANDOM_STATE = 42  # Константа зерна для случайного разделения


def main():
    '''
    Основная функция для выполнения задачи масштабирования признаков и обучения модели.
    '''

    # 1. Загрузка данных
    data_loader = DataLoader(DATASET_URL)
    iris_data = data_loader.load_data()

    if iris_data is None:
        print('Не удалось загрузить данные. Завершение работы.')
        return

    print('Данные успешно загружены.\n')
    print(iris_data.head())

    # Удаляем столбец 'Id', так как он не является признаком
    if 'Id' in iris_data.columns:
        iris_data = iris_data.drop('Id', axis=1)


    # 2. Разделение на признаки (X) и целевую переменную (y)
    X = iris_data.drop('Species', axis=1)
    y = iris_data['Species']


    # 3. Визуализация распределения признаков (до масштабирования)
    data_visualizer = DataVisualizer()
    data_visualizer.create_scatter_plot_plotly(iris_data, x_column='SepalLengthCm', y_column='SepalWidthCm', color_column='Species', title='Диаграмма рассеяния: SepalLengthCm vs SepalWidthCm (до масштабирования)')

    # Гистограммы до масштабирования
    data_visualizer.create_histograms(X, title='Гистограммы признаков до масштабирования', window_title='Гистограммы до масштабирования')


    # 4. Масштабирование признаков
    scaler = Scaler()
    numerical_columns = X.columns.tolist()  # Список числовых столбцов

    X_scaled_standard = scaler.scale_standard(X.copy(), numerical_columns)
    X_scaled_minmax = scaler.scale_min_max(X.copy(), numerical_columns)
    X_scaled_robust = scaler.scale_robust(X.copy(), numerical_columns)

    print('Признаки успешно масштабированы.\n')

    # Добавляем обратно столбец 'Species' для визуализации
    X_scaled_standard['Species'] = iris_data['Species']
    X_scaled_minmax['Species'] = iris_data['Species']
    X_scaled_robust['Species'] = iris_data['Species']


    # 5. Визуализация распределения признаков (после масштабирования)
    data_visualizer.create_scatter_plot_plotly(X_scaled_standard, x_column='SepalLengthCm', y_column='SepalWidthCm', color_column='Species', title='Диаграмма рассеяния: SepalLengthCm vs SepalWidthCm (StandardScaler)')
    data_visualizer.create_scatter_plot_plotly(X_scaled_minmax, x_column='SepalLengthCm', y_column='SepalWidthCm', color_column='Species', title='Диаграмма рассеяния: SepalLengthCm vs SepalWidthCm (MinMaxScaler)')
    data_visualizer.create_scatter_plot_plotly(X_scaled_robust, x_column='SepalLengthCm', y_column='SepalWidthCm', color_column='Species', title='Диаграмма рассеяния: SepalLengthCm vs SepalWidthCm (RobustScaler)')

    # Гистограммы после масштабирования
    data_visualizer.create_histograms(X_scaled_standard.drop('Species', axis=1), title='Гистограммы признаков после StandardScaler', window_title='Гистограммы StandardScaler')
    data_visualizer.create_histograms(X_scaled_minmax.drop('Species', axis=1), title='Гистограммы признаков после MinMaxScaler', window_title='Гистограммы MinMaxScaler')
    data_visualizer.create_histograms(X_scaled_robust.drop('Species', axis=1), title='Гистограммы признаков после RobustScaler', window_title='Гистограммы RobustScaler')


    # 6. Разделение данных на обучающий и тестовый наборы
    model_trainer = ModelTrainer(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = model_trainer.split_data(X, y)
    X_train_scaled_standard, X_test_scaled_standard, _, _ = model_trainer.split_data(X_scaled_standard.drop('Species', axis=1), y)  # Удаляем 'Species' перед обучением
    X_train_scaled_minmax, X_test_scaled_minmax, _, _ = model_trainer.split_data(X_scaled_minmax.drop('Species', axis=1), y) # Удаляем 'Species' перед обучением
    X_train_scaled_robust, X_test_scaled_robust, _, _ = model_trainer.split_data(X_scaled_robust.drop('Species', axis=1), y) # Удаляем 'Species' перед обучением


    # 7. Обучение и оценка модели на исходных данных
    model_trainer.train_model(X_train, y_train)
    accuracy_original = model_trainer.evaluate_model(X_test, y_test)
    print(f'\nТочность модели на исходных данных: {accuracy_original:.4f}')


    # 8. Обучение и оценка модели на масштабированных данных (StandardScaler)
    model_trainer.train_model(X_train_scaled_standard, y_train)
    accuracy_standard = model_trainer.evaluate_model(X_test_scaled_standard, y_test)
    print(f'Точность модели на данных, масштабированных StandardScaler: {accuracy_standard:.4f}')


    # 9. Обучение и оценка модели на масштабированных данных (MinMaxScaler)
    model_trainer.train_model(X_train_scaled_minmax, y_train)
    accuracy_minmax = model_trainer.evaluate_model(X_test_scaled_minmax, y_test)
    print(f'Точность модели на данных, масштабированных MinMaxScaler: {accuracy_minmax:.4f}')


    # 10. Обучение и оценка модели на масштабированных данных (RobustScaler)
    model_trainer.train_model(X_train_scaled_robust, y_train)
    accuracy_robust = model_trainer.evaluate_model(X_test_scaled_robust, y_test)
    print(f'Точность модели на данных, масштабированных RobustScaler: {accuracy_robust:.4f}')


if __name__ == '__main__':
    main()