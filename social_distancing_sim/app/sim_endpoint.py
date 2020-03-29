from typing import Dict, Any

from fastapi import FastAPI
from starlette.responses import FileResponse

from social_distancing_sim.sim.sim_templates import (herd_of_cats_visual_example, socially_responsible_visual_example,
                                                     herd_of_cats_stats_example, socially_responsible_stats_example,
                                                     herd_of_cats_vs_socially_responsible_stats_example)

app = FastAPI()


@app.get('/examples/herd_of_cats')
async def run_example_herd_of_cats():
    await herd_of_cats_visual_example.run()
    print(herd_of_cats_visual_example.populations[0].output_path)
    return FileResponse(herd_of_cats_visual_example.populations[0].output_gif)


@app.get('/examples/socially_responsible/')
async def run_example_socially_responsible():
    await socially_responsible_visual_example.run()
    return FileResponse(socially_responsible_visual_example.populations[0].output_gif)


@app.get('/examples/socially_responsible_multiple_runs/')
async def run_example_socially_responsible_multiple_runs() -> Dict[str, Any]:
    await herd_of_cats_stats_example.run()
    return herd_of_cats_stats_example.summary().to_dict()


@app.get('/examples/herd_of_cats_multiple_runs/')
async def run_example_herd_of_cats_multiple_runs() -> Dict[str, Any]:
    await socially_responsible_stats_example.run()
    return socially_responsible_stats_example.summary().to_dict()
