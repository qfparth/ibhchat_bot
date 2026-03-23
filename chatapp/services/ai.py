import requests


def ask_llama(prompt: str, system: str = None) -> str:
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        }
    }

    if system:
        payload["system"] = system

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
        )
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Cannot connect to AI service. Please try again later."
    except Exception as e:
        return f"Something went wrong: {str(e)}"