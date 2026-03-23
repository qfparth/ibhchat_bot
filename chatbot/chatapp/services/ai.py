import requests


def ask_llama(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )

    return res.json()["response"]
