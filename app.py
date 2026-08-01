import streamlit as st
import requests

st.set_page_config(page_title="Medikal Asistan", page_icon="🏥")
st.title("🏥 Medikal Asistan AI")

# Hugging Face model linkin
API_URL = "https://api-inference.huggingface.co/models/Mehmetalper/medikal-asistan-gguf"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
            res = requests.post(API_URL, json={"inputs": prompt})
            try:
                ans = res.json()[0]["generated_text"]
            except:
                ans = "Model şu an hazırlanıyor veya yanıt vermedi, birkaç saniye sonra tekrar dene."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
