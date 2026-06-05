import gradio as gr
from openai import OpenAI
import os, time, requests
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"

client = OpenAI(api_key="vllm-no-key", base_url=f"{VLLM_URL}/v1")

# Persona definitions: (system_prompt, temperature)
PERSONAS = {
    "Code Assistant":  ("You are an expert programmer. Answer with code examples and brief technical explanations.", 0.2),
    "Teacher":         ("You are a patient teacher. Explain everything simply with examples.", 0.5),
    "Creative Writer": ("You are a creative writer with vivid imagination. Write expressively.", 1.2),
}

# Check if vLLM server is reachable
def check_server():
    try:
        if requests.get(f"{VLLM_URL}/health", timeout=5).status_code == 200:
            return "Server Online"
    except Exception:
        pass
    return "Server Offline"

# Fetch model info from vLLM /v1/models endpoint
def get_model_info():
    try:
        data = requests.get(f"{VLLM_URL}/v1/models", timeout=5).json()
        m = data["data"][0]
        return (
            f"**Model:** `{m['id']}`\n\n"
            f"**Owned by:** {m.get('owned_by', 'vllm')}\n\n"
            f"**Server:** `{VLLM_URL}`\n\n"
            f"**API Format:** OpenAI-compatible"
        )
    except Exception:
        return "Could not fetch model info. Is the server running?"

# Export conversation history to a .txt file
def export_chat(history):
    if not history:
        return None
    lines = ["vLLM Chat Export", "=" * 40]
    for msg in history:
        role = "You" if msg["role"] == "user" else "Bot"
        lines.append(f"\n{role}: {msg['content']}")
    path = "/tmp/chat_export.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

# Main chat function with streaming — history is list of dicts (Gradio 6)
def chat(message, history, system_prompt, temperature, max_tokens):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    new_history = list(history) + [{"role": "user", "content": message}]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        stream=True
    )

    partial, tokens, start = "", 0, time.time()
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            tokens += 1
            yield new_history + [{"role": "assistant", "content": partial}]

    elapsed = time.time() - start
    final = f"{partial}\n\n---\n*{elapsed:.1f}s | ~{tokens} tokens*"
    yield new_history + [{"role": "assistant", "content": final}]

# Build the Gradio interface
with gr.Blocks(title="vLLM Chat Demo") as demo:

    gr.Markdown("# vLLM Chat Demo")
    gr.Markdown(f"Small Language Model served via **vLLM 0.9.0** | Model: `{MODEL}`")

    with gr.Tabs():

        # Chat tab
        with gr.TabItem("Chat"):

            with gr.Row():
                status_box = gr.Textbox(
                    value=check_server(),
                    label="Server Status",
                    interactive=False,
                    scale=3
                )
                gr.Button("Refresh", scale=1).click(check_server, outputs=status_box)

            gr.Markdown("### Personas")
            with gr.Row():
                btn_code     = gr.Button("Code Assistant")
                btn_teacher  = gr.Button("Teacher")
                btn_creative = gr.Button("Creative Writer")

            with gr.Accordion("Parameters", open=False):
                system_prompt = gr.Textbox(
                    value="You are a helpful assistant.",
                    label="System Prompt",
                    lines=2
                )
                with gr.Row():
                    temperature = gr.Slider(
                        0.1, 1.5, value=0.7, step=0.1,
                        label="Temperature (0.1 = precise, 1.5 = creative)"
                    )
                    max_tokens = gr.Slider(
                        64, 1024, value=512, step=64,
                        label="Max Tokens"
                    )

            chatbot = gr.Chatbot(height=400, label="Conversation")
            msg_box = gr.Textbox(placeholder="Type your message...", label="Message")

            with gr.Row():
                send_btn   = gr.Button("Send", variant="primary")
                clear_btn  = gr.Button("Clear")
                export_btn = gr.Button("Download Chat")

            export_file = gr.File(label="Chat Export", visible=False)

            # Persona buttons update system prompt and temperature
            btn_code.click(    lambda: PERSONAS["Code Assistant"],  outputs=[system_prompt, temperature])
            btn_teacher.click( lambda: PERSONAS["Teacher"],         outputs=[system_prompt, temperature])
            btn_creative.click(lambda: PERSONAS["Creative Writer"], outputs=[system_prompt, temperature])

            # Send message on button click or Enter
            send_btn.click(
                chat,
                inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens],
                outputs=chatbot
            ).then(lambda: "", outputs=msg_box)

            msg_box.submit(
                chat,
                inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens],
                outputs=chatbot
            ).then(lambda: "", outputs=msg_box)

            clear_btn.click(lambda: [], outputs=chatbot)

            export_btn.click(
                export_chat,
                inputs=chatbot,
                outputs=export_file
            ).then(lambda: gr.File(visible=True), outputs=export_file)

        # Model Info tab
        with gr.TabItem("Model Info"):

            gr.Markdown("### Model Information")
            model_info = gr.Markdown(value=get_model_info())
            gr.Button("Refresh").click(get_model_info, outputs=model_info)

            gr.Markdown("""
### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Send messages, get response |
| `/v1/models` | GET | List available models |
| `/health` | GET | Check server status |

### Architecture

**vLLM** serves the model via an OpenAI-compatible REST API.
The **client** uses the OpenAI Python SDK pointed at the vLLM server instead of `api.openai.com`.
            """)

demo.launch(server_name="0.0.0.0", server_port=7860)