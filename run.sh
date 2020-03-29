#!/usr/bin/env bash
uvicorn social_distancing_sim.app.sim_endpoint:app --reload --host 0.0.0.0
