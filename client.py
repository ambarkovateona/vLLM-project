import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# configuration
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"

client = OpenAI(
    api_key="vllm-no-key",
    base_url=f"{VLLM_URL}/v1"
)

# demo 1: simple QA
def simple_qa(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question}
        ],
        max_tokens=256,
        temperature=0.7
    )
    return response.choices[0].message.content


# demo 2: multi-turn chat with history
def multi_turn_chat():
    history = []
    questions = [
        "Hello! Who are you?",
        "Can you write code?",
        "Write a fibonacci function in Python."
    ]

    print("=== Multi-turn Chat Demo ===\n")
    for q in questions:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for human, assistant in history:
            messages.append({"role": "user",      "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": q})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        history.append((q, answer))

        print(f"You: {q}")
        print(f"Bot: {answer}\n")


# demo 3: streaming answer
def streaming_demo(question: str):
    print(f"=== Streaming Demo ===\nYou: {question}\nBot: ", end="", flush=True)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question}
        ],
        max_tokens=256,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


# main
if __name__ == "__main__":
    print("=== Demo 1: Simple Q&A ===")
    answer = simple_qa("What is Alan Turing known for?")
    print(f"Answer: {answer}\n")

    multi_turn_chat()

    streaming_demo("Explain what vLLM is in 2 sentences.")