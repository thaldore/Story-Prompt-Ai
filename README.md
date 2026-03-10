# 📖 Story Prompt AI

> **Multi-model AI story generator** — generate creative stories from a single prompt using three different AI models side by side.

---

## 🇬🇧 English

### Overview

**Story Prompt AI** is a Streamlit web application that lets you generate fictional stories from a text prompt using three AI models simultaneously:

| Model | Type | Notes |
|-------|------|-------|
| **GPT-2** | Fine-tuned transformer | Runs fully offline |
| **Gemini 1.5 Flash** | Google generative AI | Requires an API key & internet |
| **Seq2Seq (LSTM)** | Custom-trained encoder-decoder | Runs fully offline |

The app optionally translates Turkish prompts to English before generation, and translates the generated stories back to Turkish — so you can write in either language.

---

### Project Structure

```
Story-Prompt-Ai/
├── app.py               # Streamlit UI — main application entry point
├── train.py             # GPT-2 fine-tuning script
├── train_seq2seq.py     # Seq2Seq (LSTM) model definition & training script
├── dataset.json         # Prompt → Story training dataset
├── model/               # Fine-tuned GPT-2 model files (not tracked in Git)
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── vocab.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── seq2seq_model.pth    # Trained Seq2Seq weights (not tracked in Git)
├── requirements.txt     # Python dependencies
└── .gitignore
```

> ⚠️ **Model files** (`model/`, `*.pth`, `*.safetensors`) are excluded from Git because of their size. Download or train them separately (see below).

---

### Requirements

- Python 3.9+
- CUDA-capable GPU (optional, falls back to CPU)

Install dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary>Core packages used</summary>

- `streamlit`
- `transformers`
- `torch`
- `deep-translator`
- `google-generativeai`
- `tqdm`
- `datasets`

</details>

---

### Quick Start

#### 1 — Set your Gemini API key

Open `app.py` and replace the placeholder:

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

#### 2 — Prepare the models

**Option A — Train from scratch:**

```bash
# Fine-tune GPT-2
python train.py

# Train the Seq2Seq LSTM model
python train_seq2seq.py
```

**Option B — Use pre-trained weights:**  
Place your `model/` directory and `seq2seq_model.pth` file in the project root.

#### 3 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### How It Works

1. Type (or select) a story prompt.
2. Choose which models to use and adjust the GPT-2 output length.
3. Enable or disable Turkish ↔ English translation.
4. Click **Hikayeleri Oluştur** — stories from all selected models appear side by side.
5. Switch between Turkish and English outputs with the language toggle in each card.

---

### Training Details

| | GPT-2 | Seq2Seq |
|---|---|---|
| Base model | `gpt2` (HuggingFace) | Custom LSTM |
| Epochs | 10 | 10 |
| Batch size | 4 | 2 |
| Embedding dim | — | 256 |
| Hidden dim | — | 512 |
| Optimizer | AdamW (Trainer) | Adam + StepLR |
| Data | `dataset.json` | `dataset.json` |

---

### License

This project is open-source. Feel free to use, modify, and distribute it.

---
---

## 🇹🇷 Türkçe

### Genel Bakış

**Story Prompt AI**, tek bir metin girdisinden üç farklı yapay zeka modelini aynı anda kullanarak yaratıcı hikayeler üreten bir Streamlit web uygulamasıdır.

| Model | Tür | Notlar |
|-------|-----|--------|
| **GPT-2** | Fine-tune edilmiş transformer | Tamamen çevrimdışı çalışır |
| **Gemini 1.5 Flash** | Google üretken yapay zeka | API anahtarı ve internet gerektirir |
| **Seq2Seq (LSTM)** | Özel eğitilmiş encoder-decoder | Tamamen çevrimdışı çalışır |

Uygulama isteğe bağlı olarak Türkçe promptları üretim öncesinde İngilizce'ye çevirir, üretilen hikayeleri de tekrar Türkçe'ye çevirir.

---

### Proje Yapısı

```
Story-Prompt-Ai/
├── app.py               # Streamlit arayüzü — uygulamanın giriş noktası
├── train.py             # GPT-2 fine-tuning betiği
├── train_seq2seq.py     # Seq2Seq model tanımı ve eğitim betiği
├── dataset.json         # Prompt → Hikaye eğitim veri kümesi
├── model/               # Fine-tune edilmiş GPT-2 model dosyaları (Git'e eklenmez)
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── vocab.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── seq2seq_model.pth    # Eğitilmiş Seq2Seq ağırlıkları (Git'e eklenmez)
├── requirements.txt     # Python bağımlılıkları
└── .gitignore
```

> ⚠️ **Model dosyaları** (`model/`, `*.pth`, `*.safetensors`) büyük boyutları nedeniyle Git'e eklenmez. Aşağıdaki adımları takip ederek ayrıca indirip oluşturabilirsiniz.

---

### Gereksinimler

- Python 3.9 ve üzeri
- CUDA destekli GPU (isteğe bağlı, yoksa CPU kullanılır)

Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

---

### Hızlı Başlangıç

#### 1 — Gemini API anahtarını ayarla

`app.py` dosyasını açın ve aşağıdaki satırı düzenleyin:

```python
genai.configure(api_key="GEMINI_API_ANAHTARINIZ")
```

#### 2 — Modelleri hazırla

**Seçenek A — Sıfırdan eğit:**

```bash
# GPT-2 modelini fine-tune et
python train.py

# Seq2Seq LSTM modelini eğit
python train_seq2seq.py
```

**Seçenek B — Hazır ağırlıkları kullan:**  
`model/` klasörünü ve `seq2seq_model.pth` dosyasını proje kök dizinine yerleştirin.

#### 3 — Uygulamayı başlat

```bash
streamlit run app.py
```

Tarayıcınızda [http://localhost:8501](http://localhost:8501) adresini açın.

---

### Kullanım

1. Bir hikaye promptu yazın veya hazır örneklerden seçin.
2. Kullanmak istediğiniz modelleri seçin ve GPT-2 çıktı uzunluğunu ayarlayın.
3. Türkçe ↔ İngilizce çeviriyi açıp kapatın.
4. **Hikayeleri Oluştur** butonuna tıklayın — seçilen tüm modellerden hikayeler yan yana görünür.
5. Her kartta dil geçiş düğmesiyle Türkçe ve İngilizce çıktılar arasında geçiş yapın.

---

### Eğitim Detayları

| | GPT-2 | Seq2Seq |
|---|---|---|
| Temel model | `gpt2` (HuggingFace) | Özel LSTM |
| Epoch | 10 | 10 |
| Batch boyutu | 4 | 2 |
| Gömme boyutu | — | 256 |
| Gizli katman boyutu | — | 512 |
| Optimizer | AdamW (Trainer) | Adam + StepLR |
| Veri | `dataset.json` | `dataset.json` |

---

### Lisans

Bu proje açık kaynaklıdır. Özgürce kullanabilir, değiştirebilir ve dağıtabilirsiniz.
