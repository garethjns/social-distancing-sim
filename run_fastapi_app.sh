#!/bin/sh
uvicorn app.fastapi.sim_endpoint:app --reload --host 0.0.0.0
