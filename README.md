# Масштабирование признаков и обучение модели Iris <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT-License image"></a>

**Проект по дисциплине:** МДК 13.01 Основы применения методов искусственного интеллекта в программировании

**Практическое занятие №5:** Масштабирование признаков


## 1. Введение
В рамках данного практического занятия была выполнена задача масштабирования признаков датасета Iris с использованием библиотек **Pandas**, **Scikit-learn**, **Seaborn** и **Plotly**. 

Целью работы являлось *изучение влияния различных методов масштабирования на производительность модели машинного обучения* и *визуализация результатов масштабирования*.


## 2. Скриншоты выполненного задания и конспекта лекции
### 2.1. Выполненное задание
#### 2.1.1. Скрипт [main.py](src/main.py)
<p align="center">
  <img src="https://github.com/user-attachments/assets/a3334d36-121e-40d0-83f3-02ffbec0ea73" alt="main.py">
</p>

#### 2.1.2. Скрипт [data_loader.py](src/data_loader.py)
<p align="center">
  <img src="https://github.com/user-attachments/assets/ad8c0875-817f-435f-9e42-b68a2a7842c6" alt="data_loader.py">
</p>

#### 2.1.3. Скрипт [data_visualizer.py](src/data_visualizer.py)
<p align="center">
  <img src="https://github.com/user-attachments/assets/c6a68e9f-8e07-41c5-a5fa-55ad02f10743" alt="data_visualizer.py">
</p>

#### 2.1.4. Скрипт [scaler.py](src/scaler.py)
<p align="center">
  <img src="https://github.com/user-attachments/assets/46015dd5-088f-49c2-8162-67c327f979b0" alt="scaler.py">
</p>

#### 2.1.5. Скрипт [model_trainer.py](src/model_trainer.py)
<p align="center">
  <img src="https://github.com/user-attachments/assets/9988caa9-ecef-4caa-910a-00d74016e45c" alt="model_trainer.py">
</p>

### 2.2. Конспект лекции
<p align="center">
  <img src="report\lecture-notes\lecture-notes-1.jpg" alt="lecture-notes-1.jpg">
  <img src="report\lecture-notes\lecture-notes-2.jpg" alt="lecture-notes-2.jpg">
  <img src="report\lecture-notes\lecture-notes-3.jpg" alt="lecture-notes-3.jpg">
</p>


## 3. Методика и подходы
### 3.1. Методы
В ходе работы применялись следующие методы:

* **Загрузка данных:** Загрузка данных из *CSV-файла* с использованием *Pandas*.
* **Визуализация данных:**
    * **Диаграмма рассеяния (Plotly):** Для визуализации распределения признаков до и после масштабирования.
    * **Гистограммы (Matplotlib, Seaborn):** Для анализа распределения признаков до и после масштабирования.
* **Масштабирование признаков:** Применение *StandardScaler*, *MinMaxScaler* и *RobustScaler* к признакам датасета.
* **Обучение модели и оценка результатов:** Обучение логистической регрессии и оценка точности на исходных и масштабированных данных.

### 3.2. Алгоритмы
Для обработки данных и построения графиков использовались следующие алгоритмы:

* **StandardScaler:** Для стандартизации признаков.
* **MinMaxScaler:** Для масштабирования признаков в диапазоне [0, 1].
* **RobustScaler:** Для масштабирования признаков с учетом выбросов.
* **Логистическая регрессия:** Для классификации видов Iris.

### 3.3. Подходы
* **Объектно-ориентированное программирование (ООП):** Использована объектно-ориентированная парадигма для организации кода, разделение ответственности между классами `DataLoader`, `DataVisualizer`, `Scaler` и `ModelTrainer`.
* **Принципы SOLID, KISS и DRY:** Применен подход, обеспечивающий гибкость, простоту и отсутствие дублирования кода.

### 3.4. Допущения и ограничения
* Предполагается, что данные, загруженные из *CSV-файла*, корректны и не содержат ошибок.
* Использована логистическая регрессия, которая может не быть оптимальной для данного набора данных.
* Оценка производительности модели ограничена точностью (accuracy).

### 3.5. Инструменты, библиотеки и технологии
* **Python:** основной язык программирования.
* **Pandas:** для загрузки и обработки данных.
* **Scikit-learn:** для машинного обучения (масштабирование, логистическая регрессия).
* **Matplotlib:** для создания статических графиков.
* **Seaborn:** для улучшения визуализации графиков.
* **Plotly:** для создания интерактивных графиков.
* **Requests:** для загрузки данных из URL.


## 4. Результаты
### 4.1. Краткое описание данных
Данные были взяты из репозитория: *https://github.com/fcschmidt/knn_iris_dataset/blob/master/datasets/iris/Iris.csv*. Формат данных - *CSV*. Набор данных содержит информацию о цветках Iris, включая *длину чашелистика*, *ширину чашелистика*, *длину лепестка*, *ширину лепестка* и *вид*.

### 4.2. Предварительная обработка данных
* Удален столбец 'Id', как нерелевантный признак.
* Разделение данных на признаки (X) и целевую переменную (y).
* Масштабирование признаков с использованием *StandardScaler*, *MinMaxScaler* и *RobustScaler*.

### 4.3. Графики и диаграммы
#### 4.3.1. Графики до масштабирования
Графики показывают распределение *SepalLengthCm* и *SepalWidthCm* до масштабирования.

##### Гистограммы **Seaborn**
<p align="center">
  <img src="https://github.com/user-attachments/assets/fc1e60e1-c1b9-417f-9b1d-a322d66684e1" alt="Гистограмма Seaborn до масштабирования">
</p>

##### Диаграмма рассеяния **Plotly**
<p align="center">
  <img src="https://github.com/user-attachments/assets/5c481b6c-7aed-4fba-a097-912f0ddebd9c" alt="Диаграмма рассеяния Plotly SepalLengthCm vs SepalWidthCm до масштабирования">
</p>

#### 4.3.2. Графики после StandardScaler
Графики показывают распределение признаков после масштабирования *StandardScaler*.

##### Гистограммы **Seaborn**
<p align="center">
  <img src="https://github.com/user-attachments/assets/06a94d98-1881-44a5-88cb-3c08ac40e57f" alt="Гистограммы Seaborn после StandardScaler">
</p>

##### Диаграмма рассеяния **Plotly**
<p align="center">
  <img src="https://github.com/user-attachments/assets/2260415e-7169-4309-b3b7-4b4e56f64697" alt="Диаграмма рассеяния Plotly после StandardScaler">
</p>

#### 4.3.3. Графики после MinMaxScaler
Графики показывают распределение признаков после масштабирования *MinMaxScaler*.

##### Гистограммы **Seaborn**
<p align="center">
  <img src="https://github.com/user-attachments/assets/80c5cd07-80a7-4a24-ac23-2e27c302ee5e" alt="Гистограммы Seaborn после MinMaxScaler">
</p>

##### Диаграмма рассеяния **Plotly**
<p align="center">
  <img src="https://github.com/user-attachments/assets/04a947c5-cac7-4878-969a-2c429126168d" alt="Диаграмма рассеяния Plotly после MinMaxScaler">
</p>

#### 4.3.4. Графики после RobustScaler
Графики показывают распределение признаков после масштабирования *RobustScaler*.

##### Гистограммы **Seaborn**
<p align="center">
  <img src="https://github.com/user-attachments/assets/87b05927-0b31-4c5f-a93c-527d9d0f01a0" alt="Гистограммы Seaborn после RobustScaler">
</p>

##### Диаграмма рассеяния **Plotly**
<p align="center">
  <img src="https://github.com/user-attachments/assets/2b97bc28-01da-4b0e-b746-5c0f25734f9e" alt="Диаграмма рассеяния Plotly после RobustScaler">
</p>


## 5. Анализ результатов
### 5.1. Точность модели на различных данных

* **Точность модели на исходных данных:** 1.0000
* **Точность модели на данных, масштабированных StandardScaler:** 0.9667
* **Точность модели на данных, масштабированных MinMaxScaler:** 0.9000
* **Точность модели на данных, масштабированных RobustScaler:** 0.9667

<p align="center">
  <img src="https://github.com/user-attachments/assets/feb09b0f-3bc0-488e-b7ec-70ec69c1a482" alt="Точность моделей">
</p>

### 5.2. Выводы
* Масштабирование признаков влияет на производительность модели логистической регрессии.
* Разные методы масштабирования (*StandardScaler*, *MinMaxScaler*, *RobustScaler*) могут давать разные результаты.
* Визуализация данных помогает понять влияние масштабирования на распределение признаков.

### 5.3. Обсуждение возможных улучшений
* Использовать другие модели машинного обучения для классификации Iris.
* Применить другие методы оценки производительности модели (например, *F1-score*, *ROC AUC*).
* Исследовать влияние других параметров масштабирования и модели на результаты.


## 6. Заключение
В ходе данного проекта были применены навыки *масштабирования признаков*, *визуализации данных* и *обучения модели машинного обучения с использованием библиотек Python*. Это позволило изучить влияние различных методов масштабирования на производительность модели и получить полезные выводы.


## 7. Лицензия
Этот проект распространяется под лицензией MIT - смотрите файл [LICENSE](LICENSE) для деталей.


## 8. Автор
Бедин Владислав ([MindlessMuse666](https://github.com/MindlessMuse666))

* GitHub: [MindlessMuse666](https://github.com/MindlessMuse666 "Владислав: https://github.com/MindlessMuse666")
* Telegram: [@mindless_muse](t.me/mindless_muse)
* Gmail: [mindlessmuse.666@gmail.com](mindlessmuse.666@gmail.com)
