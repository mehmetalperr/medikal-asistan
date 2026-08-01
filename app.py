import streamlit as st
import requests

st.set_page_config(page_title="Medikal Asistan AI", page_icon="🏥", layout="centered")

st.title("🏥 Medikal Asistan Yapay Zeka")
st.caption("Eğitilmiş klinik değerlendirme ve tıbbi danışmanlık paneli.")

# Model Public olduğu için doğrudan ücretsiz Inference API adresi
API_URL = "https://api-inference.huggingface.co/models/Mehmetalper/medikal-asistan-v1"

def ask_model(prompt):
    formatted_prompt = f"<|im_start|>system\nSen uzman bir tıp asistanısın. Türkçe sorulan klinik durumlara ve semptomlara tıbbi kılavuzlara uygun yanıtlar ver.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Token veya header yok, tamamen public istek
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True
        }
    }
    
    response = requests.post(API_URL, json=payload, timeout=90)
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
                    elif isinstance(data, dict) and "generated_text" in data:
                        answer = data["generated_text"].replace("<|im_end|>", "").strip()
                    else:
                        answer = "Model yanıt üretti fakat format ayrıştırılamadı."
                elif res.status_code in [503, 504]:
                    answer = "⏳ **Model Uykudan Uyanıyor:** HF sunucusu modeli RAM'e yüklüyor. Lütfen 15-20 saniye sonra soruyu tekrar gönderin."
                else:
                    answer = f"⚠️ Hata Kodu ({res.status_code}): {res.text}"
            except Exception as e:
                answer = f"⚠️ Bir hata oluştu: {str(e)}"
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
