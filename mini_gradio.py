import gradio as gr

def echo(x):
    return x.upper()

print("[mini] importing ok")

demo = gr.Interface(fn=echo, inputs="text", outputs="text", title="Ping")
print("[mini] launching...")

demo.launch(server_name="127.0.0.1", server_port=7861, share=False, debug=True, show_error=True)
