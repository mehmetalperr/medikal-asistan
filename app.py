import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.set_page_config(page_title="Medikal Asistan AI", page_icon="🏥")

st.title("🏥 Medikal Asistan Yapay Zeka")
st.caption("Kendi eğitilmiş medikal çekirdek modeliniz ile klinik değerlendirme.")

# Kendi GGUF Modelini Hugging Face'ten İndirip Önbelleğe Alma
@st.cache_resource
def load_custom_model():
    # Hugging Face'teki kendi model yolun ve dosya adın
    model_path = hf_hub_download(
        repo_id="Mehmetalper/medikal-asistan-gguf",
        filename="qwen2.5-0.5b.Q4_K_M.gguf"
    )
    # CPU üzerinde hafif çalıştırma ayarları
    return Llama(
        model_path=model_path,
        n_ctx=1024,        # Bellek taşmaması için bağlam boyutunu 1024 yaptık
        n_threads=2,       # Streamlit ücretsiz CPU çekirdek sınırı
        verbose=False
    )

# Model yüklenirken ekranda göster
with st.spinner("Kendi medikal modeliniz belleğe yükleniyor... (İlk açılış 1-2 dk sürebilir)"):
    try:
        llm = load_custom_model()
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")

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
        with st.spinner("Modeliniz yanıt üretiyor..."):
            # Modelin eğitildiği ChatML formatı
            formatted_prompt = f"<|im_start|>system\nSen uzman bir tıp asistanısın. Türkçe olarak sorulan semptom ve klinik durumlara tıbbi kılavuzlara uygun yanıtlar verirsin.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            try:
                response = llm(
                    formatted_prompt,
                    max_tokens=256,
                    stop=["<|im_end|>"],
                    temperature=0.3,
                    echo=False
                )
                answer = response["choices"][0]["text"].strip()
            except Exception as e:
                answer = f"Yanıt üretilirken bir hata oluştu: {e}"
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
