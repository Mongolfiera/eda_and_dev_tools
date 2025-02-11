FROM python:3.10-slim-buster

RUN pip install --no-cache-dir explainerdashboard scikit-learn plotly==5.0.0

COPY dashboard.py ./
COPY app.py ./

RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]