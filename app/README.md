Веб сервис, построенный с использованием библиотеки streamlit.
Для запуска локально, установите requirements и в данной папке выоплните:

streamlit run analytics_app.py

Также возможен запуск в докере. Из данной папки выполнить в комндной строке:
1. docker build -t streamlitapp:latest -f Dockerfile .
2. docker run -p 8501:8501 streamlitapp:latest
