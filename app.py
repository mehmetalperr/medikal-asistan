import streamlit as st
import requests

st.set_page_config(page_title="Medikal Asistan AI", page_icon="🏥", layout="centered")

st.title("🏥 Medikal Asistan Yapay Zeka")
st.caption("Eğitilmiş klinik değerlendirme ve tıbbi danışmanlık paneli.")

# Hugging Face üzerindeki yeni safetensors modelin
API_URL = "https://api-inference.huggingface.co/models/Mehmetalper/medikal-asistan-v1"

def ask_model(prompt):
    formatted_prompt = f"<|im_start|>system\nSen uzman bir tıp asistanısın. Türkçe sorulan klinik durumlara ve semptomlara tıbbi kılavuzlara uygun yanıtlar ver.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True  # Model uykudaysa HF sunucusuna 'uyandırıp cevabı ver' talimatı gönderir
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
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
                    if isinstance(data, list) and len(data) > 0:
                        answer = data[0].get("generated_text", "").replace("<|im_end|>", "").strip()
                    else:
                        answer = "Model boş yanıt döndürdü."
                elif res.status_code == 503:
                    answer = "⏳ **Model Soğuk Başlatma Yapıyor:** Hugging Face sunucusu modelini belleğe yüklüyor. Yaklaşık 20-30 saniye sonra aynı soruyu tekrar gönderin."
                else:
                    answer = f"⚠️ HTTP Hata Kodu: {res.status_code} - {res.text}"
            except requests.exceptions.Timeout:
                answer = "⚠️ Zaman aşımı: Model yanıtı uzun sürdü, lütfen tekrar deneyin."
            except Exception as e:
                answer = f"⚠️ Bir hata oluştu: {str(e)}"
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
