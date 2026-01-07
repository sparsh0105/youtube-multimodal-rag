def get_recent_history(chat_history, n_turns=3):
    return chat_history[-2 * n_turns:]


def should_rewrite_llm(llm, chat_history, user_question, n_turns=3):
    if not chat_history:
        return False

    recent_history = get_recent_history(chat_history, n_turns)

    prompt = f"""
Conversation:
{recent_history}

User question:
{user_question}

Does the user question depend on earlier context to be fully understood?
Answer ONLY with "Yes" or "No".
"""
    response = llm.invoke(prompt).content.strip().lower()
    return response.startswith("yes")


def rewrite_question_llm(llm, chat_history, user_question, n_turns=3):
    recent_history = get_recent_history(chat_history, n_turns)

    prompt = f"""
Conversation:
{recent_history}

Rewrite the user's question so that it is fully self-contained and unambiguous.
Do NOT add new information.
Do NOT answer the question.

User question:
{user_question}

Rewritten question:
"""
    return llm.invoke(prompt).content.strip()


def get_effective_question(llm, chat_history, user_question, n_turns=3):
    if not chat_history:
        return user_question

    if should_rewrite_llm(llm, chat_history, user_question, n_turns):
        return rewrite_question_llm(
            llm, chat_history, user_question, n_turns
        )

    return user_question
