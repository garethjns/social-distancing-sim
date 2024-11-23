FROM python:3.9-buster
EXPOSE 8000

COPY app /code/app
COPY requirements.txt /code/
COPY social_distancing_sim /code/social_distancing_sim
COPY pyproject.toml /code/
WORKDIR /code
RUN pip install .[fastapi] --no-cache-dir

CMD ["uvicorn", "app.fastapi.sim_endpoint:app", "--host", "0.0.0.0"]
