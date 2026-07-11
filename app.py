import gradio as gr
from openai import OpenAI
import os, time, requests
from dotenv import load_dotenv
from database import (
    init_db, authenticate,
    create_conversation, get_user_conversations,
    get_conversation_messages, save_message,
    clear_conversation, delete_conversation,
    update_conversation_title,
    init_usage_table, log_token_usage, get_user_token_usage, get_all_users_usage
)

load_dotenv()
init_db()
init_usage_table()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"

client = OpenAI(api_key="vllm-no-key", base_url=f"{VLLM_URL}/v1")

PERSONAS = {
    "Code Assistant":  ("You are an expert programmer. Answer with code examples and brief technical explanations.", 0.2),
    "Teacher":         ("You are a patient teacher. Explain everything simply with examples.", 0.5),
    "Creative Writer": ("You are a creative writer with vivid imagination. Write expressively.", 1.2),
}


def check_server():
    try:
        if requests.get(f"{VLLM_URL}/health", timeout=5).status_code == 200:
            return "Server Online"
    except Exception:
        pass
    return "Server Offline"


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


def build_choices(convs):
    choices = [c["title"] for c in convs]
    mapping = {c["title"]: c["id"] for c in convs}
    return choices, mapping


def load_initial_data(request: gr.Request):
    username = request.username
    convs = get_user_conversations(username)

    if not convs:
        create_conversation(username)
        convs = get_user_conversations(username)

    choices, mapping = build_choices(convs)
    selected = choices[0]
    conv_id  = mapping[selected]
    messages = get_conversation_messages(conv_id)

    return gr.Radio(choices=choices, value=selected), messages, conv_id, mapping


def switch_conversation(selected_title, mapping):
    if not selected_title or not mapping:
        return [], None
    conv_id  = mapping.get(selected_title)
    messages = get_conversation_messages(conv_id)
    return messages, conv_id


def new_chat(request: gr.Request):
    username = request.username
    create_conversation(username)
    convs = get_user_conversations(username)
    choices, mapping = build_choices(convs)
    selected = choices[0]
    conv_id  = mapping[selected]
    return gr.Radio(choices=choices, value=selected), [], conv_id, mapping


def clear_current(conv_id):
    if conv_id:
        clear_conversation(conv_id)
    return []


def refresh_conv_list(request: gr.Request, conv_id):
    """Rebuild the sidebar list, keeping the current conversation selected."""
    username = request.username
    convs = get_user_conversations(username)
    choices, mapping = build_choices(convs)
    selected = next((c for c in choices if mapping[c] == conv_id), choices[0] if choices else None)
    return gr.Radio(choices=choices, value=selected), mapping


def chat(message, history, system_prompt, temperature, max_tokens, conv_id, request: gr.Request):
    username = request.username

    # Auto-generate a title from the first user message in this conversation
    if not history:
        title = message[:40] + ("..." if len(message) > 40 else "")
        existing_titles = {c["title"] for c in get_user_conversations(username) if c["id"] != conv_id}
        if title in existing_titles:
            title = f"{title} ({conv_id})"
        update_conversation_title(conv_id, title)

    save_message(username, conv_id, "user", message)

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
        stream=True,
        stream_options={"include_usage": True}
    )

    partial, tokens, start = "", 0, time.time()
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            tokens += 1
            yield new_history + [{"role": "assistant", "content": partial}]

    save_message(username, conv_id, "assistant", partial)

    if usage:
        prompt_t, completion_t = usage.prompt_tokens, usage.completion_tokens
    else:
        prompt_t, completion_t = 0, tokens
    log_token_usage(username, conv_id, prompt_t, completion_t)

    elapsed = time.time() - start
    final = (f"{partial}\n\n---\n"
             f"*{elapsed:.1f}s | prompt: {prompt_t} | completion: {completion_t} tokens*")
    yield new_history + [{"role": "assistant", "content": final}]


def get_my_usage(request: gr.Request):
    u = get_user_token_usage(request.username)
    return (f"### Moja potrosuvacka ({request.username})\n\n"
            f"| | |\n|---|---|\n"
            f"| Prompt tokens | {u['prompt']} |\n"
            f"| Completion tokens | {u['completion']} |\n"
            f"| **Vkupno** | **{u['total']}** |\n"
            f"| Broj na prasanja | {u['requests']} |")


def get_all_usage():
    rows = get_all_users_usage()
    if not rows:
        return "Nema podatoci."
    md = "### Site korisnici\n\n| Korisnik | Prompt | Completion | Vkupno | Prasanja |\n|---|---|---|---|---|\n"
    for r in rows:
        md += f"| {r['username']} | {r['prompt']} | {r['completion']} | {r['total']} | {r['requests']} |\n"
    return md


# Gradio UI
with gr.Blocks(title="vLLM Chat Demo") as demo:

    conv_id_state  = gr.State(None)
    conv_map_state = gr.State({})

    with gr.Row():
        gr.Markdown("# vLLM Chat Demo")
        gr.Button("Logout", size="sm").click(None, js="window.location.href='/logout'")
    gr.Markdown(f"Small Language Model served via **vLLM 0.9.0** | Model: `{MODEL}`")

    with gr.Tabs():

        with gr.TabItem("Chat"):
            with gr.Row():

                # Left sidebar — conversation list
                with gr.Column(scale=1, min_width=180):
                    new_chat_btn = gr.Button("+ New Chat", variant="primary")
                    conv_radio   = gr.Radio(choices=[], label="Conversations", interactive=True)

                # Main chat area
                with gr.Column(scale=4):
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
                            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1,
                                label="Temperature (0.1 = precise, 1.5 = creative)")
                            max_tokens  = gr.Slider(64, 1024, value=512, step=64,
                                label="Max Tokens")

                    chatbot = gr.Chatbot(height=400, label="Conversation")
                    msg_box = gr.Textbox(placeholder="Type your message...", label="Message")

                    with gr.Row():
                        send_btn   = gr.Button("Send", variant="primary")
                        clear_btn  = gr.Button("Clear")
                        export_btn = gr.Button("Download Chat")

                    export_file = gr.File(label="Chat Export", visible=False)

            # Persona buttons
            btn_code.click(    lambda: PERSONAS["Code Assistant"],  outputs=[system_prompt, temperature])
            btn_teacher.click( lambda: PERSONAS["Teacher"],         outputs=[system_prompt, temperature])
            btn_creative.click(lambda: PERSONAS["Creative Writer"], outputs=[system_prompt, temperature])

            # New conversation
            new_chat_btn.click(
                new_chat,
                outputs=[conv_radio, chatbot, conv_id_state, conv_map_state]
            )

            # Switch conversation
            conv_radio.change(
                switch_conversation,
                inputs=[conv_radio, conv_map_state],
                outputs=[chatbot, conv_id_state]
            )

            # Send message — then refresh sidebar so the auto-title appears
            send_btn.click(
                chat,
                inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens, conv_id_state],
                outputs=chatbot
            ).then(lambda: "", outputs=msg_box).then(
                refresh_conv_list,
                inputs=[conv_id_state],
                outputs=[conv_radio, conv_map_state]
            )

            msg_box.submit(
                chat,
                inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens, conv_id_state],
                outputs=chatbot
            ).then(lambda: "", outputs=msg_box).then(
                refresh_conv_list,
                inputs=[conv_id_state],
                outputs=[conv_radio, conv_map_state]
            )

            clear_btn.click(clear_current, inputs=conv_id_state, outputs=chatbot)

            export_btn.click(
                export_chat, inputs=chatbot, outputs=export_file
            ).then(lambda: gr.File(visible=True), outputs=export_file)

        with gr.TabItem("Model Info"):
            gr.Markdown("### Model Information")
            model_info = gr.Markdown(value=get_model_info())
            gr.Button("Refresh").click(get_model_info, outputs=model_info)
            gr.Markdown(f"""
### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Send messages, get response |
| GET  | `/v1/models` | List available models |
| GET  | `/health` | Check server status |

### How to use the API directly

```
POST {VLLM_URL}/v1/chat/completions
Content-Type: application/json

{{
  "model": "{MODEL}",
  "messages": [{{"role": "user", "content": "Hello!"}}],
  "temperature": 0.7,
  "max_tokens": 512
}}
```

### Architecture

**vLLM** serves the model via an OpenAI-compatible REST API.
The **client** uses the OpenAI Python SDK pointed at the vLLM server instead of `api.openai.com`.
            """)

        with gr.TabItem("Usage"):
            my_usage = gr.Markdown()
            gr.Button("Refresh").click(get_my_usage, outputs=my_usage)
            gr.Markdown("---")
            all_usage = gr.Markdown()
            gr.Button("Refresh All").click(get_all_usage, outputs=all_usage)
            demo.load(get_my_usage, outputs=my_usage)
            demo.load(get_all_usage, outputs=all_usage)

    demo.load(
        load_initial_data,
        outputs=[conv_radio, chatbot, conv_id_state, conv_map_state]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=authenticate,
    auth_message="Welcome to vLLM Chat Demo. Please log in."
)