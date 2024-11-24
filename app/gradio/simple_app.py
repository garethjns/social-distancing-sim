import gradio as gr

from app.gradio.single_population import run_single_population

interface = gr.Interface(
    fn=run_single_population,
    inputs=[
        gr.Text(
            value="./gradio/simple-example/",
            label="Output path",
        ),
        gr.Slider(minimum=5, maximum=1000, step=1, value=50),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0),
        gr.Slider(minimum=1, maximum=100, step=1, value=50),
        gr.Slider(minimum=1, maximum=100, step=1, value=15),
        gr.Slider(minimum=-0.0, maximum=1.0, step=0.01, value=0.06),
        gr.Slider(minimum=-0.0, maximum=1.0, step=0.01, value=0.04),
    ],
    outputs=[gr.Image(format="gif")],
)

interface.launch()
