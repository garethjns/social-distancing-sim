FROM python:3.7-buster
EXPOSE 8000

COPY . /code
WORKDIR /code
RUN pip install .
CMD ["uvicorn", "app.sim_endpoint:app", "--host", "0.0.0.0"]
