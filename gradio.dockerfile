FROM python:3.10-slim

COPY social_distancing_sim/ code/social_distancing_sim
COPY pyproject.toml /code
WORKDIR /code
RUN pip install .[gradio] --no-cache-dir
COPY app/ app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "-m", "app.gradio.sim_app"]
