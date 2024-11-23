FROM python:3.10-slim

WORKDIR /code
COPY app /code/app
COPY requirements.txt /code/
COPY social_distancing_sim /code/social_distancing_sim
COPY pyproject.toml /code/
RUN pip install .[gradio] --no-cache-dir
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app/gradio/sim_app.py"]
