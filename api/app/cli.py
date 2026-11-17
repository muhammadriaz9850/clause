from app.services.agent import ClauseAgent


def main():
    clause_agent = ClauseAgent()
    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue

            reply = clause_agent.ask(question)

            print(f"AI: {reply.answer}")
            print(f"--- searched: {reply.searches}")
            print(f"--- pages:    {[s.page for s in reply.sources]}")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        clause_agent.close()


if __name__ == "__main__":
    main()

# cd api
# uv run python -m app.cli
