"""
train.py — GPT-2 Fine-Tuning Betiği
=====================================
dataset.json içindeki prompt → hikaye çiftlerini kullanarak
HuggingFace GPT-2 modelini fine-tune eder ve ./model dizinine kaydeder.

Kullanım:
    python train.py
"""
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import json

# ── 1. Eğitim verisini yükle ────────────────────────────────────────────────
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ── 2. Prompt + hikayeyi tek bir string olarak birleştir ────────────────────
# Format: <|prompt|> ... <|story|> ... <|end|>
texts = [{"text": f"<|prompt|> {item['prompt']} <|story|> {item['story']} <|end|>"} for item in data]
dataset = Dataset.from_list(texts)

# ── 3. Tokenizer ve model tanımları ─────────────────────────────────────────
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# ── 4. Veriyi tokenize et ────────────────────────────────────────────────────
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ── 5. Eğitim hiperparametreleri ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,  # Bellek durumuna göre 2, 4 veya 8 seçilebilir
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # GPU varsa karma hassasiyetle eğitimi hızlandırır
)

# ── 6. Dil modelleme için veri düzenleyici ────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ── 7. Trainer nesnesini oluştur ──────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ── 8. Eğitimi başlat ─────────────────────────────────────────────────────────
trainer.train()

# ── 9. Model ve tokenizer'ı diske kaydet ──────────────────────────────────────
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
