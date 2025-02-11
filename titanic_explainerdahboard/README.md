1. Построим контейнер
`$ docker build -t explainerdashboard .`
 
2. Запустим его!
`$ docker run -p 9050:9050 explainerdashboard`

3. Смотрим что получилось:
http://localhost:9050/

Скачать образ:
`docker pull mongolfierad/titanc_explainerdashboard:latest`
