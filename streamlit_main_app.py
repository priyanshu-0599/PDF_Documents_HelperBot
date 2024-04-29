from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
from typing import Set


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


st.header("PDF Documents Helper-Bot")
query = st.chat_input(placeholder="Enter your query")

if (
    "chat_answers_history" not in st.session_state
    and "user_query_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_query_history"] = []
    st.session_state["chat_history"] = []

if query:
    with st.spinner("Generating Response"):
        generated_response = run_llm(
            query=query, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [
                f"{docs.metadata['source']}\\page:{docs.metadata['page']}"
                for docs in generated_response["source_documents"]
            ]
        )
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )
        st.session_state.chat_history.append((query, generated_response["answer"]))
        st.session_state.user_query_history.append(query)
        st.session_state.chat_answers_history.append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_query_history"],
    ):
        st.chat_message('user').write(user_query)
        st.chat_message('assistant').write(generated_response)
