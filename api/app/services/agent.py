from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app.schemas import AgentReply, Source, ToolCall
from app.core.config import get_settings
from app.prompts.loader import get_prompt
from app.llm.models import get_chat_model
from app.rag.vectorstore import get_vector_store
from app.tools.knowledge.handbook import file_search


class ClauseAgent:
    def __init__(self):
        self.settings = get_settings()
        self.store = get_vector_store(settings=self.settings)

        self.agent = create_agent(
            model=get_chat_model(settings=self.settings),
            tools=[
                file_search(store=self.store, top_k=self.settings.top_k),
            ],
            system_prompt=get_prompt(
                name="support/system",
                version=self.settings.system_prompt_version
            )
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
        self.store.client.close()
