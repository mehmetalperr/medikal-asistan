import streamlit as st
import requests

st.set_page_config(page_title="Medikal Asistan AI", page_icon="🏥")

st.title("🏥 Medikal Asistan Yapay Zeka")
st.caption("Eğitilmiş yerli tıp çekirdek modeli ile klinik değerlendirme paneli.")

# Hugging Face API Ayarları
API_URL = "https://api-inference.huggingface.co/models/MehmetAlper/medikal-asistan-gguf"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girdisi
if prompt := st.chat_input("Klinik durumu veya semptomları yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Tıbbi literatür taranıyor..."):
            formatted_prompt = f"<|im_start|>system\nSen uzman bir tıp asistanısın.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            output = query({
                "inputs": formatted_prompt,
                "parameters": {"max_new_tokens": 512, "temperature": 0.3}
            })
            
            if isinstance(output, list) and len(output) > 0:
                answer = output[0].get("generated_text", "").split("<|im_start|>assistant\n")[-1]
            else:
                answer = "Model yanıt veriyor, lütfen birkaç saniye sonra tekrar deneyin."
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
