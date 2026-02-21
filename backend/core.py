import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents  import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = PineconeVectorStore(index_name=os.getenv('INDEX_NAME'), embedding=embeddings) 

model = init_chat_model(model="models/gemini-3-flash-preview",model_provider="google_genai", temperature=0.2)

@tool(response_format="content_and_artifact")
def retrive_context(query: str) -> str:
    """Retrieve relevant context from the vector store based on the query about LangChain."""
    retrived_documents = vectorstore.as_retriever().invoke(query, k=4)

    """Serialize the retrieved documents into a string format."""
    context = "\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in retrived_documents)
    
    return context, retrived_documents

def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to  answer the given query and retrieved context.
    
    Args:
        query (str): The input query about LangChain.
    Returns:
        Dict[str, Any]: A dictionary containing 
            - answer : the LLM response 
            - context : the retrieved context.
    """
    # Crete the agent with retrival tool and LLM
    system_prompt = (
        "You are a helpful assistant that provides accurate and concise answers to questions about LangChain. "
        "You have access to a tool that retrieves relevant context from a vector store based on the query. "
        "Use the tool to find relevant context for answering the query. "
        "Always cite the sources of the retrieved context in your answer. "
        "If the context does not contain relevant information, say so.")

    agent = create_agent(
        tools=[retrive_context],
        model=model,
        system_prompt=system_prompt
    )

    # Build message list
    message = [{"role": "user", "content": query}]

    # Invoke the agent with the query and get the response
    response = agent.invoke({"messages": message})
    answer = response["messages"][-1].content

    # Extract the context from the tool message
    context_docs = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact") and msg.artifact is not None:
            # the artifact contains the retrieved documents from the tool
            if isinstance(msg.artifact, list):
                context_docs.extend(msg.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    query = "What are the main features of LangChain?"
    result = run_llm(query)
    print("Answer:", result["answer"])
    