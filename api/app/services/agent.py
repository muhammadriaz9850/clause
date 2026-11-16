from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage


SYSTEM_PROMPT = """
You are a helpful Clause AI agent.
"""

llm = init_chat_model(
    model="qwen3.5:4b",
    model_provider="ollama",
    temperature=0.5
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=SYSTEM_PROMPT
)


query = input("You: ")
# messages = [
#     {"role": "user", "content": query}
# ]
# result = agent.invoke({"messages": messages})
result = agent.invoke({"messages": [HumanMessage(query)]})
print(result["messages"][-1].text)

# cd api
# uv run python -m app.services.agent
