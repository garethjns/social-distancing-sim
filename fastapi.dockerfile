FROM python:3.9-slim
EXPOSE 8000

COPY social_distancing_sim/ code/social_distancing_sim
COPY pyproject.toml /code
WORKDIR /code
RUN pip install .[fastapi] --no-cache-dir
COPY app/ app

CMD ["uvicorn", "app.fastapi.sim_endpoint:app", "--host", "0.0.0.0"]
