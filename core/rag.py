from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import SYSTEM_PROMPT, USER_PROMPT 

def format_docs(docs):
    formatted = []
    for doc in docs:
        start = doc.metadata.get("start")
        end = doc.metadata.get("end")
        content = doc.page_content

        formatted.append(
            f"[Time: {start}-{end}s]\n{content}"
        )
    return "\n\n".join(formatted)


def answer_question(llm, retriever, question):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT.format(
                question=question,
                context=context
            )
        )
    ]

    response = llm.invoke(messages)
    return response.content
