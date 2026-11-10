from services.ingestion import ingest
from services.chat import RAGService


def main():
    rag_service = RAGService()
    try:
        query = input("You: ").strip()
        result = rag_service.ask(query)
        print(f"AI: {result.answer}")
    except ValueError as e:
        print(f"Error: {e}")
    except (KeyboardInterrupt, EOFError):
        print()
    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        rag_service.close()


if __name__ == "__main__":
    # ingest()
    main()
