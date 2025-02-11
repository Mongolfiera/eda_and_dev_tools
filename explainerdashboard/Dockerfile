FROM python:3.10-slim-buster

RUN pip install --no-cache-dir pandas numpy joblib dill explainerdashboard scikit-learn plotly==5.0.0

RUN mkdir -p app/data app/model

COPY eda.py /app/eda.py
COPY model.py /app/model.py
COPY dashboard.py /app/dashboard.py
COPY app.py /app/app.py

WORKDIR /app

RUN python eda.py
RUN python model.py
RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]