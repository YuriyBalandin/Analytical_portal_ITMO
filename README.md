# Аналитический портал

Данный проект выполнен в рамках соревнования [Learning Analytics Competition](https://ods.ai/competitions/learning-analytics) для поступления в магистратуру ИТМО по Инженерии машинного обучения. 

Разработанный сервис направлен на помощь студентам в учебе. Он помогает следить за текущими оценками, показывает рейтинг студента в сравнении с другими, чем мотивирует его быть лучше, а также с помощью ml модели предсказывает вероятность не сдать дисциплину в будущем. Предсказывая вероятность не сдать дисциплину, сервис должен мотивировать студента заранее задумываться о потенциально проблемной дисциплине и готовиться к ней более тщательно.

Демонстрацию работы сервиса можно посмотреть [здесь](https://drive.google.com/file/d/1O4tyrwpJ08Mo9ZXqssNQuRpYf5EY1QLt/view?usp=sharing).

![Снимок экрана (192)](https://user-images.githubusercontent.com/61317465/178121357-af64f228-afed-45d2-9415-f6bf0ec53f1b.png)

В сервисе используется catboost для предсказания вероятности не сдать дисциплину. Модель может быть переобучена на новых данных с использованием представленного интерфейса. Сервис также поддерживает запуск из docker, что позволяет избавиться от проблем с запуском на любом устройстве. 

В папке app представлен код и данные для работы сервиса, а также Dockerfile для запуска из докера. В папке notebooks расположен ноутбук с EDA и обучением модели. В папке utils находятся скрипты для сбора данных и обучения модели, которая может быть переобучена локально с использованием этих скриптов. В requirements.txt приведены необходимые зависимости для локального запуска. 
