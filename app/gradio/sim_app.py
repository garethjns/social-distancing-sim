import gradio as gr

from app.gradio.single_population import run_single_population

with gr.Blocks(theme=gr.themes.Glass()) as interface:
    with gr.Tab("Simple example"):
        simple_population_inputs = []
        with gr.Row():
            with gr.Column():
                gr.Markdown("Simulation parameters")
                simple_population_inputs.extend(
                    [
                        gr.Text(
                            value="./gradio/simple-example/",
                            label="Output path",
                        ),
                        gr.Slider(
                            minimum=5,
                            maximum=100,
                            step=1,
                            value=20,
                            label="N steps",
                        ),
                        gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=1.0,
                            label="Test rate",
                            interactive=False,
                        ),
                    ]
                )

                gr.Markdown("Community parameters")
                simple_population_inputs.extend(
                    [
                        gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=3,
                            label="N communities",
                        ),
                        gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=10,
                            label="Mean community size",
                        ),
                    ]
                )
            simple_population_outputs = [gr.Image(format="gif")]
        simple_button = gr.Button("Run")

    with gr.Tab("Single pop"):
        single_population_inputs = []
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Simulation parameters")
                        single_population_inputs.extend(
                            [
                                gr.Text(
                                    value="./gradio/example/",
                                    label="Output path",
                                    key="name",
                                ),
                                gr.Slider(
                                    minimum=5,
                                    maximum=1000,
                                    step=1,
                                    value=50,
                                    label="N steps",
                                ),
                                gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.25,
                                    label="Test rate",
                                ),
                            ]
                        )

                        gr.Markdown("Population parameters")
                        single_population_inputs.extend(
                            [
                                gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=50,
                                    label="N communities",
                                ),
                                gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=15,
                                    label="Mean community size",
                                ),
                                gr.Slider(
                                    minimum=-0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.06,
                                    label="Connection rate within communities",
                                ),
                                gr.Slider(
                                    minimum=-0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.04,
                                    label="Connection rate between communities",
                                ),
                            ]
                        )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("Disease parameters")
                    single_population_inputs.extend(
                        [
                            gr.Slider(
                                minimum=-0.0,
                                maximum=1.0,
                                step=0.001,
                                value=0.005,
                                label="Virulence",
                            ),
                            gr.Slider(
                                minimum=-0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.99,
                                label="Recovery rate",
                            ),
                            gr.Slider(
                                minimum=-0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.66,
                                label="Immunity after recovery",
                            ),
                            gr.Slider(
                                minimum=-0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.1,
                                label="Immunity decay rate",
                            ),
                        ]
                    )

                    gr.Markdown("Healthcare parameters")
                    single_population_inputs.extend(
                        [
                            gr.Slider(
                                minimum=1,
                                maximum=1000,
                                step=1,
                                value=200,
                                label="Capacity",
                            ),
                            gr.Slider(
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                value=0.5,
                                label="Max penalty when over capacity",
                            ),
                        ]
                    )

        single_population_outputs = [gr.Image(format="gif")]
        single_pop_button = gr.Button("Run")

    simple_button.click(
        run_single_population,
        inputs=simple_population_inputs,
        outputs=simple_population_outputs,
    )
    single_pop_button.click(
        run_single_population,
        inputs=single_population_inputs,
        outputs=single_population_outputs,
    )


interface.launch()
