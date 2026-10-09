## THIS FILE COMPLETES THE AGENT.
## A small agent that answers university IT questions using the RAG tool,
## producing grounded answers that do not invent facts such as phone numbers.

from rag import answer


def run(query):
    print("=" * 40)
    print("QUESTION:", query)
    print("ANSWER:", answer(query))
    print()


if __name__ == "__main__":
    # The case RAG is designed to fix: a phone number that is NOT in the knowledge base.
    run("What exact phone number should I call for the University IT Service Desk?")
    run("What should I do if I forget to release my print job within 24 hours?")
