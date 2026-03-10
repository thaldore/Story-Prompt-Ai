"""
train_seq2seq.py — Seq2Seq (LSTM) Model Tanımı ve Eğitim Betiği
================================================================
LSTM tabanlı encoder-decoder mimarisi kullanarak prompt → hikaye
eşleştirmesi öğrenen özel bir Seq2Seq modeli tanımlar ve eğitir.

Dışa aktarılan bileşenler (app.py tarafından kullanılır):
    Tokenizer, Encoder, Decoder, Seq2Seq, generate,
    DEVICE, EMBEDDING_DIM, HIDDEN_DIM

Kullanım:
    python train_seq2seq.py
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# --------------------
# Ayarlar ve sabitler
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_EPOCHS = 10
MAX_LEN = 100
SOS_token = 0
EOS_token = 1
PAD_token = 2

# --------------------
# Basit Tokenizer
# --------------------
class Tokenizer:
    def __init__(self):
        self.word2idx = {"<SOS>": SOS_token, "<EOS>": EOS_token, "<PAD>": PAD_token}
        self.idx2word = {SOS_token: "<SOS>", EOS_token: "<EOS>", PAD_token: "<PAD>"}
        self.vocab_count = {}
        self.vocab_size = 3

    def fit_on_texts(self, texts):
        """
        Veri kümesindeki tüm kelimeleri tarayarak kelime hazinesi oluşturur.

        Args:
            texts: Kelimelere ayrılacak metin dizelerinin listesi veya yinelenebiliri.
        """
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def text_to_seq(self, text):
        """
        Metni token ID'leri dizisine dönüştürür; başına SOS, sonuna EOS ekler.

        Args:
            text (str): Dönüştürülecek giriş metni.

        Returns:
            list[int]: Başında SOS_token, sonunda EOS_token bulunan token ID listesi.
        """
        seq = [self.word2idx.get(word, self.word2idx["<PAD>"]) for word in text.lower().split()]
        seq = [SOS_token] + seq + [EOS_token]
        return seq

    def seq_to_text(self, seq):
        """
        Token ID dizisini okunabilir metne geri dönüştürür.

        Args:
            seq (list[int]): Token ID listesi (SOS/EOS/PAD token'ları atlanır).

        Returns:
            str: Boşluklarla birleştirilmiş kelimelerden oluşan metin.
        """
        words = []
        for idx in seq:
            if idx == EOS_token:
                break
            if idx == SOS_token or idx == PAD_token:
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

# --------------------
# Dataset sınıfı
# --------------------
class StoryDataset(Dataset):
    def __init__(self, data, tokenizer):
        """
        Veriyi ve tokenizer'ı alarak PyTorch Dataset'i başlatır.

        Args:
            data (list[dict]): Her biri 'prompt' ve 'story' anahtarı içeren sözlük listesi.
            tokenizer (Tokenizer): Metinleri token ID'lerine dönüştüren tokenizer nesnesi.
        """
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        """Veri kümesindeki örnek sayısını döndürür."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Verilen indeksteki prompt ve hikayeyi token ID tensörlerine dönüştürerek döndürür.

        Args:
            idx (int): Veri kümesindeki örnek indeksi.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (giriş token tensörü, hedef token tensörü).
        """
        prompt = self.data[idx]['prompt']
        story = self.data[idx]['story']

        input_seq = self.tokenizer.text_to_seq(prompt)
        target_seq = self.tokenizer.text_to_seq(story)

        return torch.tensor(input_seq), torch.tensor(target_seq)

def collate_fn(batch):
    """
    DataLoader için özel harmanlama fonksiyonu.
    Farklı uzunluktaki giriş ve hedef dizilerini aynı batch içinde
    işleyebilmek için PAD token'ı ile doldurur.
    """

    inputs, targets = zip(*batch)
    inputs_lens = [len(seq) for seq in inputs]
    targets_lens = [len(seq) for seq in targets]

    max_input_len = max(inputs_lens)
    max_target_len = max(targets_lens)

    padded_inputs = torch.full((len(inputs), max_input_len), PAD_token)
    padded_targets = torch.full((len(targets), max_target_len), PAD_token)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tgt)] = tgt

    return padded_inputs.to(DEVICE), padded_targets.to(DEVICE), inputs_lens, targets_lens

# --------------------
# Model mimarisi
# --------------------
class Encoder(nn.Module):
    """Giriş metnini sabit boyutlu bir bağlam vektörüne (gizli durum + hücre durumu) kodlar."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        return hidden, cell

class Decoder(nn.Module):
    """Encoder'dan gelen bağlam vektörünü kullanarak çıktı dizisini (hikayeyi) adım adım oluşturur."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """Encoder ve Decoder'ı tek bir modelde birleştirir; teacher forcing destekler."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        İleri geçiş: encoder çıktısından başlayarak hedef diziyi üretir.
        teacher_forcing_ratio: her adımda gerçek hedef token'ı kullanma olasılığı.
        """
        batch_size = src.size(0)
        max_len = trg.size(1)
        vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_lengths)

        input_token = trg[:,0]  # İlk token genellikle <SOS>

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

# --------------------
# Eğitim fonksiyonu
# --------------------
def train(model, dataloader, optimizer, criterion, epoch):
    """
    Modeli bir epoch boyunca eğitir.
    Her batch sonrasında kayıp, gradyan normu ve öğrenme oranını loglar.
    """
    print(f"Using device: {DEVICE}, CUDA Available: {torch.cuda.is_available()}")
    model.train()
    epoch_loss = 0

    for i, (src, trg, src_lens, trg_lens) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        output = model(src, src_lens, trg)
        
        output_dim = output.shape[-1]
        output = output[:,1:].reshape(-1, output_dim)
        trg = trg[:,1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()

        # Gradyan normunu hesapla ve logla
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        epoch_loss += loss.item()

        # Adam optimizörünün ilk parametre grubundan anlık öğrenme oranını al
        lr = optimizer.param_groups[0]['lr']

        # Her 20 batch'te bir eğitim istatistiklerini yazdır
        if i % 20 == 0:
            print({
                'loss': loss.item(),
                'grad_norm': total_norm,
                'learning_rate': lr,
                'epoch': epoch + (i / len(dataloader))
            })


    return epoch_loss / len(dataloader)

# --------------------
# Metin üretme (inference) fonksiyonu
# --------------------
def generate(model, tokenizer, prompt, max_len=MAX_LEN):
    """
    Eğitilmiş modeli kullanarak verilen prompt'a karşılık bir hikaye üretir.
    Model değerlendirme moduna alınır; gradyan hesaplanmaz.
    """
    model.eval()
    with torch.no_grad():
        input_seq = tokenizer.text_to_seq(prompt)
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(DEVICE)
        input_len = [len(input_seq)]

        hidden, cell = model.encoder(input_tensor, input_len)

        input_token = torch.tensor([SOS_token]).to(DEVICE)

        generated_indices = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)
            if top1.item() == EOS_token:
                break
            generated_indices.append(top1.item())
            input_token = top1

        return tokenizer.seq_to_text(generated_indices)

# --------------------
# Ana program
# --------------------
def main():
    """
    Ana eğitim döngüsü:
    Veriyi yükler, tokenizer'ı oluşturur, modeli eğitir,
    ağırlıkları kaydeder ve örnek bir metin üretimi yapar.
    """
    # ── Veri yükleme ──────────────────────────────────────────────────────────
    with open("dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [item['prompt'] for item in data]
    stories = [item['story'] for item in data]

    # ── Tokenizer'ı oluştur ve tüm metinlere fit et ───────────────────────────
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(prompts + stories)

    dataset = StoryDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    decoder = Decoder(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  # Her 3 epoch'ta öğrenme oranını yarıya indir

    # ── Eğitim döngüsü ────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        loss = train(model, dataloader, optimizer, criterion, epoch)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {loss:.4f}")
        scheduler.step()  # Epoch sonunda öğrenme oranını güncelle
        print(f"Updated learning rate: {optimizer.param_groups[0]['lr']}")

    # ── Eğitim sonrası modeli kaydet ──────────────────────────────────────────
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, "seq2seq_model.pth")
    print("Model kaydedildi: seq2seq_model.pth")

    # ── Örnek metin üretimi (inference) ───────────────────────────────────────
    test_prompt = "Story about a lonely mountain climber"
    generated_story = generate(model, tokenizer, test_prompt)
    print("\nPrompt:", test_prompt)
    print("Generated story:", generated_story)


if __name__ == "__main__":
    main()
