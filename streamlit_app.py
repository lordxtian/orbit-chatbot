import streamlit as st
from chatbot import classify_intent, extract_entities, get_response

st.title("🤖 orBIT Chatbot")
st.markdown("Ask about BSIT, BSCS, or EMC programs!")

user_input = st.text_input("Your question:")

if user_input:
    intent = classify_intent(user_input)
    entities = extract_entities(user_input)
    response = get_response(intent, user_input)

    st.markdown(f"**Intent:** {intent}")
    if entities:
        st.markdown(f"**Entities:** {', '.join(entities)}")
    st.markdown(f"**Response:** {response}")
