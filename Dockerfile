FROM python:3.7-buster
EXPOSE 8000

COPY . /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD ["uvicorn", "social_distancing_sim.app.sim_endpoint:app", "--host", "0.0.0.0"]
