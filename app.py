import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.set_page_config(page_title="Medikal Asistan AI", page_icon="🏥")

st.title("🏥 Medikal Asistan Yapay Zeka")
st.caption("Eğitilmiş yerli tıp çekirdek modeli ile klinik değerlendirme paneli.")

# Modeli indir ve önbelleğe al
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="MehmetAlper/medikal-asistan-gguf",
        filename="qwen2.5-0.5b.Q4_K_M.gguf"
    )
    return Llama(model_path=model_path, n_ctx=2048, n_threads=2)

with st.spinner("Medikal model yükleniyor..."):
    llm = load_model()

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

    # Model yanıtı üretme
    with st.chat_message("assistant"):
        formatted_prompt = f"<|im_start|>system\nSen uzman bir tıp asistanısın. Tıbbi sorulara net ve doğru cevaplar verirsin.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = llm(formatted_prompt, max_tokens=512, stop=["<|im_end|>"], echo=False, temperature=0.3)
        answer = response["choices"][0]["text"].strip()
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
