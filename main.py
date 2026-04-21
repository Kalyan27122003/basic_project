from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState
from llm import get_llm

llm = get_llm("groq")


def ask_llm(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}


builder = StateGraph(MessagesState)
builder.add_node("ask_llm", ask_llm)

builder.set_entry_point("ask_llm")
builder.set_finish_point("ask_llm")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage(content="Explain RAG in simple terms")]
    })

    print(result["messages"][-1].content)