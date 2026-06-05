# vLLM Chat Demo

A small language model served via **vLLM** with an **OpenAI-compatible API**, accessible through a **Gradio** chat interface.

---

## Architecture

```
+----------------------------------+         +---------------------------+
|         Google Colab             |         |      Local Machine        |
|   (NVIDIA T4 GPU - free tier)    |         |                           |
|                                  | HTTP    |  app.py   -> Gradio UI    |
|  vLLM 0.9.0 (port 8000)         |<------->|  client.py -> Python demos|
|  Qwen2.5-0.5B-Instruct model     |         |                           |
|  ngrok tunnel                    |         +---------------------------+
+----------------------------------+
```

### Request Flow

```
User types a message in Gradio
        |
        v
OpenAI SDK builds the messages list (system + history + new message)
        |
        v
HTTP POST -> https://xxxx.ngrok-free.app/v1/chat/completions
        |
        v
ngrok tunnel -> vLLM server (localhost:8000)
        |
        v
Qwen2.5-0.5B-Instruct generates response token by token
        |
        v
Response streams back -> Gradio displays it in real time
```

---

## Tech Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| Inference Server| vLLM 0.9.0                        |
| Language Model  | Qwen/Qwen2.5-0.5B-Instruct        |
| API Format      | OpenAI-compatible REST API        |
| Tunnel          | ngrok                             |
| Chat Interface  | Gradio                            |
| GPU             | Google Colab T4 (CUDA 12.8)       |

---

## Project Structure

```
vllm-project/
├── colab_server.ipynb   <- Run in Google Colab (server setup)
├── app.py               <- Gradio chat interface (run locally)
├── client.py            <- Python demo scripts (run locally)
├── requirements.txt     <- Local dependencies
└── README.md
```

---

## Setup

### 1. Start the server (Google Colab)

Open `colab_server.ipynb` in [Google Colab](https://colab.research.google.com):

1. Set runtime: **Runtime -> Change runtime type -> T4 GPU**
2. Run all cells in order
3. Copy the ngrok public URL from the last cell

### 2. Configure local environment

Create a `.env` file in the project root:

```
VLLM_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the interface

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Features

### Chat Interface (app.py)

- **Personas** — one-click presets that change the system prompt and temperature
  - Code Assistant (temperature 0.2)
  - Teacher (temperature 0.5)
  - Creative Writer (temperature 1.2)
- **System Prompt** — customize the model's behavior
- **Temperature slider** — control response creativity (0.1 to 1.5)
- **Max Tokens slider** — control response length (64 to 1024)
- **Server Status** — live check of the vLLM server health
- **Response metadata** — shows response time and token count
- **Download Chat** — export the conversation as a .txt file
- **Model Info tab** — displays model details from the `/v1/models` endpoint

### Python Demos (client.py)

- **Demo 1** — Simple Q&A: single question, single answer
- **Demo 2** — Multi-turn chat: conversation with full history across multiple turns
- **Demo 3** — Streaming: response printed token by token in real time

---

## Why vLLM?

vLLM is a high-throughput inference engine for language models. Key advantages:

- **PagedAttention** — efficient GPU memory management
- **OpenAI-compatible API** — drop-in replacement, same SDK
- **Streaming support** — token-by-token response delivery
- **Open source** — full control, no external API costs
