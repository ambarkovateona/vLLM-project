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

CSS = """
footer { display: none !important; }

.chatbot-area ::-webkit-scrollbar { width: 0 !important; }
.chatbot-area { scrollbar-width: none !important; }

button.primary {
    background: #000000 !important;
    border-color: #000000 !important;
    color: #ffffff !important;
}
button.primary:hover {
    background: #222222 !important;
    border-color: #222222 !important;
}

button.secondary {
    border-color: #000000 !important;
    color: #000000 !important;
}
button.secondary:hover { background: #f4f4f4 !important; }

.delete-btn button {
    background: #ffffff !important;
    border: 1px solid #dc2626 !important;
    color: #dc2626 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    width: 100% !important;
}
.delete-btn button:hover { background: #fef2f2 !important; }

.selected-conv-label p {
    font-size: 0.75rem !important;
    padding: 2px 4px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

input[type="radio"]    { accent-color: #000000 !important; }
input[type="range"]    { accent-color: #000000 !important; }
input[type="checkbox"] { accent-color: #000000 !important; }

.tabs button.selected,
.tab-nav button.selected {
    border-color: #000000 !important;
    color: #000000 !important;
}
.tabs button:hover,
.tab-nav button:hover { color: #000000 !important; }

input:focus,
textarea:focus,
.block:focus-within,
.wrap:focus-within {
    border-color: #000000 !important;
    box-shadow: 0 0 0 2px rgba(0,0,0,0.08) !important;
}

.accordion-header { color: #000000 !important; }
"""


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


def build_choices(convs):
    choices = [c["title"] for c in convs]
    mapping = {c["title"]: c["id"] for c in convs}
    return choices, mapping


def get_welcome(username):
    hour = int(time.strftime("%H"))
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 18 else "Good evening")
    return f"### {greeting}, {username}!\nHow can I help you today?"


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
    welcome  = get_welcome(username) if not messages else ""
    label    = f"Selected: **{selected}**"
    return gr.Radio(choices=choices, value=selected), messages, conv_id, mapping, welcome, label


def switch_conversation(selected_title, mapping):
    if not selected_title or not mapping:
        return [], None, ""
    conv_id  = mapping.get(selected_title)
    messages = get_conversation_messages(conv_id)
    label    = f"Selected: **{selected_title}**"
    return messages, conv_id, label


def new_chat(request: gr.Request):
    username = request.username
    create_conversation(username)
    convs = get_user_conversations(username)
    choices, mapping = build_choices(convs)
    selected = choices[0]
    conv_id  = mapping[selected]
    label    = f"Selected: **{selected}**"
    return gr.Radio(choices=choices, value=selected), [], conv_id, mapping, get_welcome(username), label


def delete_current_conv(conv_id, mapping, request: gr.Request):
    if not conv_id:
        return gr.Radio(), [], None, mapping, "", ""
    username = request.username
    delete_conversation(conv_id)
    convs = get_user_conversations(username)
    if not convs:
        create_conversation(username)
        convs = get_user_conversations(username)
    choices, new_mapping = build_choices(convs)
    selected    = choices[0]
    new_conv_id = new_mapping[selected]
    messages    = get_conversation_messages(new_conv_id)
    welcome     = get_welcome(username) if not messages else ""
    label       = f"Selected: **{selected}**"
    return gr.Radio(choices=choices, value=selected), messages, new_conv_id, new_mapping, welcome, label


def clear_current(conv_id):
    if conv_id:
        clear_conversation(conv_id)
    return []


def refresh_conv_list(request: gr.Request, conv_id):
    username = request.username
    convs = get_user_conversations(username)
    choices, mapping = build_choices(convs)
    selected = next((c for c in choices if mapping[c] == conv_id), choices[0] if choices else None)
    label    = f"Selected: **{selected}**" if selected else ""
    return gr.Radio(choices=choices, value=selected), mapping, label


def chat(message, history, system_prompt, temperature, max_tokens, conv_id, request: gr.Request):
    username = request.username
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
        model=MODEL, messages=messages,
        max_tokens=int(max_tokens), temperature=float(temperature),
        stream=True, stream_options={"include_usage": True}
    )

    partial, tokens, start, usage = "", 0, time.time(), None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            tokens  += 1
            yield new_history + [{"role": "assistant", "content": partial}]

    save_message(username, conv_id, "assistant", partial)
    prompt_t     = usage.prompt_tokens     if usage else 0
    completion_t = usage.completion_tokens if usage else tokens
    log_token_usage(username, conv_id, prompt_t, completion_t)

    elapsed = time.time() - start
    final = f"{partial}\n\n---\n*{elapsed:.1f}s | prompt: {prompt_t} | completion: {completion_t} tokens*"
    yield new_history + [{"role": "assistant", "content": final}]


def get_my_usage(request: gr.Request):
    u = get_user_token_usage(request.username)
    return (f"### My Usage ({request.username})\n\n"
            f"| | |\n|---|---|\n"
            f"| Prompt tokens | {u['prompt']} |\n"
            f"| Completion tokens | {u['completion']} |\n"
            f"| **Total** | **{u['total']}** |\n"
            f"| Requests | {u['requests']} |")


with gr.Blocks(title="Qwen Chat") as demo:

    conv_id_state  = gr.State(None)
    conv_map_state = gr.State({})

    gr.Markdown("# Qwen Chat")

    with gr.Tabs():

        with gr.TabItem("Chat"):
            with gr.Row():

                with gr.Column(scale=1, min_width=180):
                    new_chat_btn = gr.Button("+ New Chat", variant="primary")
                    gr.Markdown("---")
                    conv_radio   = gr.Radio(choices=[], label="Conversations", interactive=True)
                    gr.Markdown("---")
                    selected_label = gr.Markdown("", elem_classes="selected-conv-label")
                    delete_btn     = gr.Button("Delete selected", size="sm", elem_classes="delete-btn")
                    gr.Markdown("---")
                    gr.Button("Log out", size="sm").click(None, js="window.location.href='/logout'")

                with gr.Column(scale=4):
                    with gr.Row():
                        status_box = gr.Textbox(
                            value=check_server(), label="Server Status",
                            interactive=False, scale=3
                        )
                        gr.Button("Refresh", scale=1).click(check_server, outputs=status_box)

                    welcome_box = gr.Markdown("")

                    gr.Markdown("### Personas")
                    with gr.Row():
                        btn_code     = gr.Button("Code Assistant")
                        btn_teacher  = gr.Button("Teacher")
                        btn_creative = gr.Button("Creative Writer")

                    with gr.Accordion("Parameters", open=False):
                        system_prompt = gr.Textbox(
                            value="You are a helpful assistant.",
                            label="System Prompt", lines=2
                        )
                        with gr.Row():
                            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1,
                                label="Temperature (0.1 = precise, 1.5 = creative)")
                            max_tokens  = gr.Slider(64, 1024, value=512, step=64,
                                label="Max Tokens")

                    chatbot = gr.Chatbot(height=400, label="Conversation")
                    msg_box = gr.Textbox(placeholder="Type your message...", label="Message")

                    with gr.Row():
                        send_btn  = gr.Button("Send", variant="primary")
                        stop_btn  = gr.Button("Stop")
                        clear_btn = gr.Button("Clear")

            btn_code.click(    lambda: PERSONAS["Code Assistant"],  outputs=[system_prompt, temperature])
            btn_teacher.click( lambda: PERSONAS["Teacher"],         outputs=[system_prompt, temperature])
            btn_creative.click(lambda: PERSONAS["Creative Writer"], outputs=[system_prompt, temperature])

            new_chat_btn.click(new_chat,
                outputs=[conv_radio, chatbot, conv_id_state, conv_map_state, welcome_box, selected_label])

            delete_btn.click(delete_current_conv,
                inputs=[conv_id_state, conv_map_state],
                outputs=[conv_radio, chatbot, conv_id_state, conv_map_state, welcome_box, selected_label])

            conv_radio.change(switch_conversation,
                inputs=[conv_radio, conv_map_state],
                outputs=[chatbot, conv_id_state, selected_label])

            # Store original streaming events separately so stop_btn can cancel them
            chat_event = send_btn.click(
                chat,
                inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens, conv_id_state],
                outputs=chatbot
            )
            chat_event.then(lambda: "", outputs=msg_box
            ).then(lambda: "", outputs=welcome_box
            ).then(refresh_conv_list, inputs=[conv_id_state],
                   outputs=[conv_radio, conv_map_state, selected_label])

            submit_event = msg_box.submit(
                chat,
                inputs=[msg_box, chatbot, system_prompt, temperature, max_tokens, conv_id_state],
                outputs=chatbot
            )
            submit_event.then(lambda: "", outputs=msg_box
            ).then(lambda: "", outputs=welcome_box
            ).then(refresh_conv_list, inputs=[conv_id_state],
                   outputs=[conv_radio, conv_map_state, selected_label])

            stop_btn.click(fn=None, cancels=[chat_event, submit_event])
            clear_btn.click(clear_current, inputs=conv_id_state, outputs=chatbot)

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

### Architecture
**vLLM** serves the model via an OpenAI-compatible REST API.
The **client** uses the OpenAI Python SDK pointed at the vLLM server instead of `api.openai.com`.
            """)

        with gr.TabItem("Usage"):
            my_usage = gr.Markdown()
            gr.Button("Refresh").click(get_my_usage, outputs=my_usage)
            demo.load(get_my_usage, outputs=my_usage)

    demo.load(load_initial_data,
        outputs=[conv_radio, chatbot, conv_id_state, conv_map_state, welcome_box, selected_label])

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=authenticate,
    auth_message="Welcome to Qwen Chat. Please log in.",
    css=CSS
)