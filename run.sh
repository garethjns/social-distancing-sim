#!/bin/sh
uvicorn app.sim_endpoint:app --reload --host 0.0.0.0
