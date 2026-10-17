from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()


def main():
    print("Testing TypeSafe Jev API...")

    try:
        with TypeSafeClient(model="jev-latest") as client:
            response = client.system_one(
                state={
                    "memory_candidate": (
                        "I prefer Python for backend development."
                    ),
                    "existing_memories": [
                        "The user prefers Python for backend development."
                    ],
                    "scope": "user",
                },
                questions={
                    "action": Choice(
                        instructions=(
                            "Choose the most appropriate memory action. "
                            "Consider whether an equivalent memory already "
                            "exists in the same scope."
                        ),
                        criteria={
                            "STORE": "Store a genuinely new memory.",
                            "IGNORE": "An equivalent memory already exists.",
                            "UPDATE": "The existing memory needs correction.",
                            "MERGE": "Combine complementary memories.",
                            "SUPERSEDE": "Replace an outdated memory.",
                            "CONFLICT": "The memories contradict each other.",
                            "UNCERTAIN": "Insufficient information to decide.",
                        },
                    ),
                    "confidence": Score(
                        instructions=(
                            "How strong is the evidence supporting "
                            "your selected action?"
                        ),
                        criteria=[
                            "Weak evidence",
                            "Moderate evidence",
                            "Strong evidence",
                        ],
                    ),
                    "needs_review": Noul(
                        instructions=(
                            "Does this decision require human or additional "
                            "reasoning review before changing canonical memory?"
                        ),
                    ),
                },
            )

        print("\nPASS: Jev API request succeeded.")
        print(f"Model: {response.model}")

        print("\nDecision:")
        print(response.choices["action"].choice)

        print("\nEvidence score:")
        print(response.scores["confidence"].score)

        print("\nReview probability:")
        print(response.nouls["needs_review"].noul)

    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()