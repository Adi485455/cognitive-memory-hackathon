import os

from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()


def test_groq():
    print("\n[1/2] Testing Groq...")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("FAIL: GROQ_API_KEY is missing.")
        return False

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: GROQ_OK",
                }
            ],
            max_completion_tokens=20,
        )

        answer = response.choices[0].message.content
        print("PASS: Groq API request succeeded.")
        print(f"Response: {answer}")
        return True

    except Exception as exc:
        print(f"FAIL: Groq request failed ({type(exc).__name__}).")
        print("Check the API key, model availability, and account limits.")
        return False


def test_gemini_embeddings():
    print("\n[2/2] Testing Gemini Embeddings...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("FAIL: GEMINI_API_KEY is missing.")
        return False

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents="Testing embeddings for a cognitive memory system.",
        )

        if not response.embeddings:
            print("FAIL: Gemini returned no embeddings.")
            return False

        vector = response.embeddings[0].values

        print("PASS: Gemini embedding request succeeded.")
        print(f"Vector dimensions: {len(vector)}")
        print(f"Sample values: {vector[:5]}")
        return True

    except Exception as exc:
        print(
            f"FAIL: Gemini request failed ({type(exc).__name__})."
        )
        print("Check the API key, model availability, and account limits.")
        return False


if __name__ == "__main__":
    results = {
        "Groq": test_groq(),
        "Gemini Embeddings": test_gemini_embeddings(),
    }

    print("\n========== API TEST SUMMARY ==========")
    for service, passed in results.items():
        print(f"{service}: {'PASS' if passed else 'FAIL'}")

    if not all(results.values()):
        raise SystemExit(1)