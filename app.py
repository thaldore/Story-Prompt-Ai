import streamlit as st

st.set_page_config(layout="wide")  # Geniş ekran modu

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from deep_translator import GoogleTranslator
import google.generativeai as genai
import os
import random
from train_seq2seq import Tokenizer, Encoder, Decoder, Seq2Seq, generate, DEVICE, EMBEDDING_DIM, HIDDEN_DIM

# Örnek prompt'lar
EXAMPLE_PROMPTS = [
    "Story about a sentient ancient monument that slowly awakens",
    "Story about an AI",
    "Story about a magical library that changes its contents."
]

# Model ve tokenizer'ı yükle
@st.cache_resource(show_spinner="Modeller yükleniyor...")
def load_models():
    try:
        # GPT-2 Modeli
        gpt_model_dir = os.path.abspath("./model")
        if not os.path.exists(gpt_model_dir):
            raise FileNotFoundError(f"GPT-2 model dizini bulunamadı: {gpt_model_dir}")
            
        gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_dir, local_files_only=True)
        gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_dir, local_files_only=True)
        gpt_model.eval()
        
        # Gemini Modeli
        genai.configure(api_key="Gemini_Api_Key")  # gemini api key
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Seq2Seq Modeli
        seq2seq_checkpoint = torch.load("seq2seq_model.pth", map_location=DEVICE)
        seq2seq_tokenizer = seq2seq_checkpoint["tokenizer"]

        encoder = Encoder(seq2seq_tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
        decoder = Decoder(seq2seq_tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
        seq2seq_model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
        seq2seq_model.load_state_dict(seq2seq_checkpoint["model_state_dict"])
        seq2seq_model.eval()
        
        return {
            "gpt": {"tokenizer": gpt_tokenizer, "model": gpt_model},
            "gemini": {"model": gemini_model},
            "seq2seq": {"tokenizer": seq2seq_tokenizer, "model": seq2seq_model}
        }
    except Exception as e:
        st.error(f"Modeller yüklenirken hata oluştu: {str(e)}")
        return None

# Modelleri yükle
models = load_models()

# Uygulama arayüzü
st.title("📖 Çoklu Model Hikaye Üretici")
st.markdown("**Tek bir prompt ile üç farklı modelden hikaye üretin**")

# İki sütunlu layout
col_input, col_examples = st.columns([3, 1])

# Session state'de prompt yoksa rastgele bir örnek seç
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = random.choice(EXAMPLE_PROMPTS)

with col_input:
    # Kullanıcıdan Türkçe prompt al
    turkish_prompt = st.text_area(
        "Hikaye için bir başlangıç yazın:",
        value=st.session_state.current_prompt,
        placeholder="Örnek: 'Bir sihirbazın asistanı olarak çalışmaya başladım ve...'",
        height=100,
        key="user_prompt"
    )

with col_examples:
    st.markdown("### Hazır Prompt'lar")
    selected_example = st.selectbox(
        "Örneklerden seçin:",
        EXAMPLE_PROMPTS,
        index=EXAMPLE_PROMPTS.index(st.session_state.current_prompt) if st.session_state.current_prompt in EXAMPLE_PROMPTS else 0,
        label_visibility="collapsed"
    )
    
    if selected_example != st.session_state.current_prompt:
        st.session_state.current_prompt = selected_example
        st.rerun()

# Ayarlar
col1, col2, col3 = st.columns(3)
with col1:
    max_length = st.slider(
        "GPT-2 için metin uzunluğu:",
        min_value=50,
        max_value=500,
        value=150,
        help="GPT-2 modeli için üretilecek maksimum token sayısı"
    )
with col2:
    translate = st.checkbox(
        "Çeviri yap (Türkçe → İngilizce → Model → Türkçe)",
        value=True,
        help="Prompt'u İngilizce'ye çevirip sonucu tekrar Türkçe'ye çevirir"
    )
with col3:
    selected_models = st.multiselect(
        "Kullanılacak Modeller",
        ["GPT-2", "Gemini", "Seq2Seq"],
        default=["GPT-2", "Gemini", "Seq2Seq"]
    )

# Session state ile çıktıları ve dil seçimlerini sakla
if 'outputs' not in st.session_state:
    st.session_state.outputs = {
        'gpt': {'tr': None, 'en': None},
        'gemini': {'tr': None, 'en': None},
        'seq2seq': {'tr': None, 'en': None}
    }
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = {
        'gpt': 'tr',
        'gemini': 'tr',
        'seq2seq': 'tr'
    }


def generate_stories():
    if not turkish_prompt.strip():
        st.warning("Lütfen bir hikaye başlangıcı yazın!")
        return
    
    # Çeviri işlemi
    if translate:
        with st.spinner("Prompt İngilizce'ye çevriliyor..."):
            try:
                en_prompt = GoogleTranslator(source='tr', target='en').translate(turkish_prompt)
            except Exception as e:
                st.error(f"Çeviri hatası: {str(e)}")
                return
    else:
        en_prompt = turkish_prompt

    # GPT-2 Hikayesi
    if "GPT-2" in selected_models:
        with st.spinner("GPT-2 hikaye oluşturuyor..."):
            try:
                if models and models["gpt"]["model"]:
                    inputs = models["gpt"]["tokenizer"].encode(en_prompt, return_tensors="pt")
                    with torch.no_grad():
                        outputs = models["gpt"]["model"].generate(
                            inputs,
                            max_length=max_length,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            pad_token_id=models["gpt"]["tokenizer"].eos_token_id,
                            do_sample=True,
                            top_p=0.95,
                            top_k=50,
                            temperature=0.7,
                        )
                    gpt_en_text = models["gpt"]["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                    st.session_state.outputs['gpt']['en'] = gpt_en_text
                    
                    if translate:
                        with st.spinner("GPT-2 çıktısı çevriliyor..."):
                            try:
                                gpt_tr_text = GoogleTranslator(source='en', target='tr').translate(gpt_en_text)
                                st.session_state.outputs['gpt']['tr'] = gpt_tr_text
                            except Exception as e:
                                st.error(f"Çeviri hatası: {str(e)}")
                                st.session_state.outputs['gpt']['tr'] = "Çeviri başarısız oldu"
                    else:
                        st.session_state.outputs['gpt']['tr'] = gpt_en_text
            except Exception as e:
                st.error(f"GPT-2 hatası: {str(e)}")

    # Gemini Hikayesi
    if "Gemini" in selected_models:
        with st.spinner("Gemini hikaye oluşturuyor..."):
            try:
                if models and models["gemini"]["model"]:
                    response = models["gemini"]["model"].generate_content(
                        f"Write a creative story based on this prompt: {en_prompt}. "
                        "The story should be detailed and engaging."
                    )
                    gemini_en_text = response.text
                    st.session_state.outputs['gemini']['en'] = gemini_en_text
                    
                    if translate:
                        with st.spinner("Gemini çıktısı çevriliyor..."):
                            try:
                                gemini_tr_text = GoogleTranslator(source='en', target='tr').translate(gemini_en_text)
                                st.session_state.outputs['gemini']['tr'] = gemini_tr_text
                            except Exception as e:
                                st.error(f"Çeviri hatası: {str(e)}")
                                st.session_state.outputs['gemini']['tr'] = "Çeviri başarısız oldu"
                    else:
                        st.session_state.outputs['gemini']['tr'] = gemini_en_text
            except Exception as e:
                st.error(f"Gemini hatası: {str(e)}")

    # Seq2Seq Hikayesi
    if "Seq2Seq" in selected_models:
        with st.spinner("Seq2Seq hikaye oluşturuyor..."):
            try:
                if models and models["seq2seq"]["model"]:
                    seq2seq_en_text = generate(models["seq2seq"]["model"], models["seq2seq"]["tokenizer"], en_prompt)
                    st.session_state.outputs['seq2seq']['en'] = seq2seq_en_text
                    
                    if translate:
                        with st.spinner("Seq2Seq çıktısı çevriliyor..."):
                            try:
                                seq2seq_tr_text = GoogleTranslator(source='en', target='tr').translate(seq2seq_en_text)
                                st.session_state.outputs['seq2seq']['tr'] = seq2seq_tr_text
                            except Exception as e:
                                st.error(f"Çeviri hatası: {str(e)}")
                                st.session_state.outputs['seq2seq']['tr'] = "Çeviri başarısız oldu"
                    else:
                        st.session_state.outputs['seq2seq']['tr'] = seq2seq_en_text
            except Exception as e:
                st.error(f"Seq2Seq hatası: {str(e)}")

if st.button("Hikayeleri Oluştur", type="primary"):
    generate_stories()

# Hikayeleri göster
if "GPT-2" in selected_models and st.session_state.outputs['gpt']['tr'] is not None:
    col_gpt, col_gemini, col_seq2seq = st.columns(3)
    
    with col_gpt:
        with st.container(border=True, height=500):
            st.subheader("GPT-2 Hikayesi")
            
            # Dil seçimi
            gpt_lang = st.radio(
                "Dil Seçimi",
                ["Türkçe", "İngilizce"],
                index=0 if st.session_state.current_lang['gpt'] == 'tr' else 1,
                key="gpt_lang",
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.session_state.current_lang['gpt'] = 'tr' if gpt_lang == "Türkçe" else 'en'
            
            # Çıktıyı göster
            st.markdown("---")
            if st.session_state.current_lang['gpt'] == 'tr':
                st.write(st.session_state.outputs['gpt']['tr'])
            else:
                st.write(st.session_state.outputs['gpt']['en'])

    # Gemini Hikayesi
    if "Gemini" in selected_models and st.session_state.outputs['gemini']['tr'] is not None:
        with col_gemini:
            with st.container(border=True, height=500):
                st.subheader("Gemini Hikayesi")
                
                # Dil seçimi
                gemini_lang = st.radio(
                    "Dil Seçimi",
                    ["Türkçe", "İngilizce"],
                    index=0 if st.session_state.current_lang['gemini'] == 'tr' else 1,
                    key="gemini_lang",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                st.session_state.current_lang['gemini'] = 'tr' if gemini_lang == "Türkçe" else 'en'
                
                # Çıktıyı göster
                st.markdown("---")
                if st.session_state.current_lang['gemini'] == 'tr':
                    st.write(st.session_state.outputs['gemini']['tr'])
                else:
                    st.write(st.session_state.outputs['gemini']['en'])

    # Seq2Seq Hikayesi
    if "Seq2Seq" in selected_models and st.session_state.outputs['seq2seq']['tr'] is not None:
        with col_seq2seq:
            with st.container(border=True, height=500):
                st.subheader("Seq2Seq Hikayesi")
                
                # Dil seçimi
                seq2seq_lang = st.radio(
                    "Dil Seçimi",
                    ["Türkçe", "İngilizce"],
                    index=0 if st.session_state.current_lang['seq2seq'] == 'tr' else 1,
                    key="seq2seq_lang",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                st.session_state.current_lang['seq2seq'] = 'tr' if seq2seq_lang == "Türkçe" else 'en'
                
                # Çıktıyı göster
                st.markdown("---")
                if st.session_state.current_lang['seq2seq'] == 'tr':
                    st.write(st.session_state.outputs['seq2seq']['tr'])
                else:
                    st.write(st.session_state.outputs['seq2seq']['en'])

# Yan bilgi paneli
with st.sidebar:
    st.header("⚙️ Model Bilgileri")
    
    with st.expander("GPT-2", expanded=False):
        st.caption("- Fine-tuned model")
        st.caption("- Çevrimdışı çalışır")
        st.caption("- Kısa çıktılar")
    
    with st.expander("Gemini", expanded=False):
        st.caption("- Google'ın modeli")
        st.caption("- Online çalışır")
        st.caption("- Yaratıcı çıktılar")
        
    with st.expander("Seq2Seq", expanded=False):
        st.caption("- Özel eğitilmiş model")
        st.caption("- Çevrimdışı çalışır")
        st.caption("- Veri kümesine özgü çıktılar")
    
    st.divider()
    with st.expander("Kılavuz", expanded=False):
        st.caption("1. Hikaye başlangıcı yaz")
        st.caption("2. Modelleri seç")
        st.caption("3. Ayarları yap")
        st.caption("4. Oluştur butonuna bas")
        st.caption("5. Çıktıları karşılaştır")
    
    st.caption("Story Generator with ai models")