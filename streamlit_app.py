import streamlit as st
import requests, sys, time


def fetch_messages():
    try:
        response = requests.get('http://localhost:5000/get_messages')

        if response.status_code == 200:
            return response.json()["messages"]
        else:
            return []
    except Exception as e:
        st.error(f"Failed to fetch messages: {str(e)}")
        return []


def display_message(message):
    role = message.get("role", "system")
    content = message.get("content", "")
    color = message.get("color", "gray")
    alignment = "left" if role == "user" else "right"

    st.markdown(
        f'<div style="text-align: {alignment}; background-color: {color}; color: white; padding: 10px; '
        f'border-radius: 5px; margin: 5px;">{content}</div>',
        unsafe_allow_html=True
    )


def main():
    st.title('LLM Chat Interface')
    st.markdown("""
    ### Instructions:
    - Use the terminal to input messages
    - Type `/mode chat` for regular chat mode
    - Type `/mode rag` for RAG mode with document context
    """)

    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []

    while True:
        messages = fetch_messages()
        new_messages = messages[len(st.session_state.displayed_messages):]

        if new_messages:
            for message in new_messages:
                display_message(message)
            st.session_state.displayed_messages = messages

        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()

