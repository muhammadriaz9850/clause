from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

from app.services.chat import RAGService


class ClauseAgent:
    def __init__(self):
        self.rag = RAGService()

        @tool(response_format="content_and_artifact")
        def search_handbook(query: str):
            """Search the Clause shop handbook: refunds, delivery times and fees,
            payments, cash on delivery, warranty, exchanges, tracking, loyalty, support hours.
            Use a short query like "refund window electronics"."""

            results = self.rag.store.similarity_search_with_score(
                query=query, k=self.rag.settings.top_k
            )
            if not results:
                return "Nothing found.", []

            texts = []
            sources = []
            for doc, score in results:
                texts.append(doc.page_content)
                sources.append({"page": doc.metadata.get("page"), "score": round(score, 4)})
            text = "\n\n---\n\n".join(texts)

            return text, sources

        self.agent = create_agent(
            model=self.rag.llm,
            tools=[search_handbook],
            system_prompt=self.rag.settings.system_prompt,
        )

    def ask(self, question: str) -> dict:
        result = self.agent.invoke({"messages": [HumanMessage(question)]})
        messages = result["messages"]

        searches = []   # queries the LLM chose
        sources = []    # pages Qdrant returned
        for message in messages:
            # 1. The LLM asked for a tool -> save its search query
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    searches.append(call["args"]["query"])

            # 2. The tool answered -> save its sources (the artifact)
            if isinstance(message, ToolMessage):
                if message.artifact:
                    for source in message.artifact:
                        sources.append(source)

        answer = messages[-1].text
        return {"answer": answer, "searches": searches, "sources": sources}

    def close(self):
        self.rag.close()


if __name__ == "__main__":
    clause_agent = ClauseAgent()
    try:
        while True:
            question = input("You: ").strip()
            reply = clause_agent.ask(question)
            print(f"AI_: {reply["answer"]}")
            print(f"--- searched: {reply["searches"]}")
    finally:
        clause_agent.close()


# cd api
# uv run python -m app.services.agent
