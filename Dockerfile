FROM python:3.9.12

RUN pip install --upgrade pip

WORKDIR /App
COPY . /App
COPY .env ./

RUN pip install -r requirements.txt

RUN apt-get update

EXPOSE 8995

CMD ["streamlit", "run", "interview_app.py", "--server.port", "8995"]