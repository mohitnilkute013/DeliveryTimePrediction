FROM python:3.8-alpine
COPY . /app
WORKDIR /app
# RUN apk update
# RUN apk add make automake gcc g++ subversion python3-dev
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD python app.py
