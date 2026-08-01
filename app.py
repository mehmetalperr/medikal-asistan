import streamlit as st
import requests

st.set_page_config(page_title="Medikal Asistan AI", page_icon="🏥", layout="centered")

st.title("🏥 Medikal Asistan Yapay Zeka")
st.caption("Eğitilmiş klinik değerlendirme ve tıbbi danışmanlık paneli.")

# Hugging Face Güncellenmiş Router/Inference Endpoint'i
API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"

def ask_model(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    
    # OpenAI uyumlu Chat Completion formatı (HF Inference destekler)
    payload = {
        "model": "Mehmetalper/medikal-asistan-v1",
        "messages": [
            {"role": "system", "content": "Sen uzman bir tıp asistanısın. Türkçe sorulan klinik durumlara ve semptomlara tıbbi kılavuzlara uygun yanıtlar ver."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    
    # Timeout süresini 90 saniye yapıyoruz ki soğuk başlatmada düşmesin
    response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    return response

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Klinik durumu veya semptomları yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Tıbbi değerlendirme yapılıyor..."):
            try:
                res = ask_model(prompt)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["choices"][0]["message"]["content"]
                elif res.status_code in [503, 504]:
                    answer = "⏳ **Model Soğuk Başlatma Yapıyor:** Hugging Face sunucusu modelini belleğe alıyor. Lütfen 20 saniye sonra tekrar deneyin."
                else:
                    answer = f"⚠️ HF Bağlantı Hatası: {res.status_code} - {res.text}"
            except requests.exceptions.ConnectionError:
                answer = "⚠️ **Ağ Hatası:** Streamlit sunucusu Hugging Face'e erişemedi. Sayfayı yenileyip 5 saniye sonra tekrar deneyin."
            except requests.exceptions.Timeout:
                answer = "⚠️ **Zaman Aşımı:** Yanıt süresi aşıldı. Lütfen tekrar deneyin."
            except Exception as e:
                answer = f"⚠️ Bir hata oluştu: {str(e)}"
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
