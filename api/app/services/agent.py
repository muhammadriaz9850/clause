from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

from app.schemas import AgentReply, Source, ToolCall
from app.services.chat import RAGService


class ClauseAgent:
    def __init__(self):
        self.rag = RAGService()

        @tool(response_format="content_and_artifact")
        def search_handbook(query: str) -> tuple[str, list[Source]]:
            """Search the Clause shop handbook: refunds, delivery times and fees,
            payments, cash on delivery, warranty, exchanges, tracking, loyalty, support hours.
            Use a short query like "refund window electronics"."""

            results = self.rag.store.similarity_search_with_score(
                query=query, k=self.rag.settings.top_k
            )
            if not results:
                return "Nothing found.", []

            text = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
            sources = [
                Source(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4),
                )
                for doc, score in results
            ]
            # content -> goes to the LLM; artifact (sources) -> stays with us
            return text, sources

        self.agent = create_agent(
            model=self.rag.llm,
            tools=[search_handbook],
            system_prompt=self.rag.settings.system_prompt,
        )

    def ask(self, question: str) -> AgentReply:
        result = self.agent.invoke({"messages": [HumanMessage(question)]})
        messages = result["messages"]

        tool_calls: list[ToolCall] = []  # tools the LLM chose to call
        sources: list[Source] = []       # chunks those tools returned
        for message in messages:
            # 1. The LLM asked for a tool -> record name + args
            if isinstance(message, AIMessage):
                tool_calls.extend(
                    ToolCall(name=call["name"], args=call["args"]) for call in message.tool_calls
                )

            # 2. The tool answered -> collect its sources (the artifact)
            if isinstance(message, ToolMessage) and message.artifact:
                sources.extend(message.artifact)

        return AgentReply(answer=messages[-1].text, tool_calls=tool_calls, sources=sources)

    def close(self):
        self.rag.close()
