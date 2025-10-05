# Środowisko Fine-Tuningu Modeli z LM Studio

Kompletne środowisko do fine-tuningu modeli językowych zintegrowane z LM Studio. Zawiera wszystkie niezbędne narzędzia do przygotowania danych, treningu modeli i ich ewaluacji.

## 🚀 Szybki Start

### 1. Instalacja

```bash
# Sklonuj lub pobierz repozytorium
git clone <repo-url>
cd fine_tuning_sample

# Zainstaluj zależności
pip install -r requirements.txt

# Skonfiguruj środowisko
python scripts/setup_environment.py
```

### 2. Przygotowanie danych

```bash
# Utwórz przykładowy dataset
python scripts/data_preparation.py create-sample --output data/train.jsonl --num-samples 100

# Waliduj dane
python scripts/data_preparation.py validate --input data/train.jsonl

# Podziel na train/val/test
python scripts/data_preparation.py split --input data/train.jsonl
```

### 3. Fine-tuning

```bash
# Uruchom fine-tuning
python scripts/fine_tune.py --config configs/training_config.yaml

# Lub z własnymi parametrami
python scripts/fine_tune.py --data data/train.jsonl --epochs 5 --lr 1e-5
```

### 4. Ewaluacja

```bash
# Testuj model z LM Studio
python scripts/lm_studio_client.py test --model "twój-model" --prompt "Test prompt"

# Ewaluuj na datasetcie
python scripts/lm_studio_client.py evaluate --model "twój-model" --dataset data/test.jsonl --output results.json

# Analizuj wyniki
python scripts/model_evaluation.py evaluate --results results.json --output-dir output/evaluation
```

## 📁 Struktura Projektu

```
fine_tuning_sample/
├── configs/                 # Pliki konfiguracyjne
│   ├── training_config.yaml # Konfiguracja treningu
│   └── lm_studio_config.json # Konfiguracja LM Studio
├── data/                    # Dane treningowe
│   ├── raw/                 # Surowce dane
│   └── processed/           # Przetworzone dane
├── models/                  # Modele
│   ├── base/                # Modele bazowe
│   └── fine_tuned/          # Wytrenowane modele
├── scripts/                 # Skrypty pomocnicze
│   ├── setup_environment.py # Konfiguracja środowiska
│   ├── fine_tune.py         # Główny skrypt treningu
│   ├── lm_studio_client.py  # Klient LM Studio
│   ├── data_preparation.py  # Przygotowanie danych
│   └── model_evaluation.py  # Ewaluacja modeli
├── output/                  # Wyniki
│   ├── checkpoints/         # Checkpointy treningu
│   └── logs/                # Logi
├── logs/                    # Logi systemowe
├── requirements.txt         # Zależności Python
└── README.md               # Dokumentacja
```

## ⚙️ Konfiguracja

### Plik `configs/training_config.yaml`

Główny plik konfiguracyjny zawiera:

- **Model**: nazwa, ścieżka, parametry
- **Trening**: epochs, batch size, learning rate
- **LoRA**: konfiguracja adaptacji
- **Optymalizacja pamięci**: fp16, gradient checkpointing
- **Dane**: ścieżki do datasetów
- **Logowanie**: wandb, tensorboard

### Plik `configs/lm_studio_config.json`

Konfiguracja integracji z LM Studio:

- **API**: URL, timeout, retry
- **Modele**: lista dostępnych modeli
- **Formaty**: szablony instrukcji i odpowiedzi

## 📊 Format Danych

Dane treningowe w formacie JSONL:

```json
{"instruction": "Wyjaśnij czym jest AI", "response": "Sztuczna inteligencja to..."}
{"instruction": "Jak gotować makaron?", "response": "1. Zagotuj wodę..."}
```

### Konwersja z innych formatów

```bash
# CSV → JSONL
python scripts/data_preparation.py convert-csv --input data.csv --output data.jsonl

# TXT → JSONL
python scripts/data_preparation.py convert-txt --input data.txt --output data.jsonl
```

## 🔧 Narzędzia

### 1. Przygotowanie Danych (`scripts/data_preparation.py`)

- Konwersja formatów (CSV, TXT → JSONL)
- Walidacja datasetów
- Filtrowanie według długości
- Podział na train/val/test
- Tworzenie przykładowych danych

### 2. Fine-Tuning (`scripts/fine_tune.py`)

- Trening z LoRA
- Optymalizacja pamięci
- Monitorowanie z TensorBoard
- Zapis checkpointów
- Integracja z Transformers

### 3. Klient LM Studio (`scripts/lm_studio_client.py`)

- Komunikacja z API LM Studio
- Testowanie modeli
- Ewaluacja na datasetach
- Generowanie odpowiedzi

### 4. Ewaluacja (`scripts/model_evaluation.py`)

- Metryki BLEU i ROUGE
- Analiza czasów odpowiedzi
- Wizualizacja wyników
- Porównanie modeli
- Generowanie raportów

## 🎯 Przykłady Użycia

### Fine-tuning modelu Llama-2

```bash
# 1. Przygotuj dane
python scripts/data_preparation.py create-sample --output data/llama_data.jsonl --num-samples 500

# 2. Skonfiguruj model w configs/training_config.yaml
# model:
#   name: "llama-2-7b-chat"
#   path: "./models/"

# 3. Uruchom trening
python scripts/fine_tune.py --config configs/training_config.yaml

# 4. Przetestuj model
python scripts/lm_studio_client.py test --model "llama-2-7b-chat"
```

### Ewaluacja wielu modeli

```bash
# Ewaluuj modele
python scripts/lm_studio_client.py evaluate --model "model-1" --dataset data/test.jsonl --output results1.json
python scripts/lm_studio_client.py evaluate --model "model-2" --dataset data/test.jsonl --output results2.json

# Porównaj wyniki
python scripts/model_evaluation.py compare --compare-results results1.json results2.json --compare-output comparison.csv
```

### Optymalizacja pamięci

```yaml
# W configs/training_config.yaml
memory_optimization:
  gradient_checkpointing: true
  fp16: false
  bf16: true
  use_8bit_optimizer: true

lora:
  enabled: true
  r: 8  # Mniejsza wartość = mniej pamięci
```

## 📈 Monitorowanie

### TensorBoard

```bash
# Uruchom TensorBoard
tensorboard --logdir output/logs

# Otwórz http://localhost:6006
```

### Logi

Logi zapisywane w:
- `logs/training/fine_tune.log` - logi treningu
- `output/logs/` - logi TensorBoard
- `logs/` - logi systemowe

## 🐛 Rozwiązywanie Problemów

### Błąd: "CUDA out of memory"

```yaml
# Zmniejsz batch size
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

# Włącz gradient checkpointing
memory_optimization:
  gradient_checkpointing: true
```

### Błąd: "Model not found"

1. Sprawdź ścieżkę w `configs/training_config.yaml`
2. Upewnij się, że model jest w katalogu `models/`
3. Sprawdź format modelu (GGUF dla LM Studio)

### Błąd: "LM Studio connection failed"

1. Uruchom LM Studio
2. Załaduj model
3. Sprawdź port 1234
4. Sprawdź konfigurację w `configs/lm_studio_config.json`

## 📚 Dokumentacja Dodatkowa

### Metryki Ewaluacji

- **BLEU**: Jakość generowanego tekstu (0-1)
- **ROUGE**: Pokrycie informacji (0-1)
- **Czas odpowiedzi**: Wydajność modelu
- **Stosunek długości**: Porównanie z oczekiwaniami

### Optymalizacja Wydajności

1. **LoRA**: Zmniejsza liczbę parametrów do trenowania
2. **Gradient Checkpointing**: Oszczędza pamięć
3. **Mixed Precision**: fp16/bf16 dla szybszego treningu
4. **DataLoader**: Optymalizacja ładowania danych

## 🤝 Wkład w Projekt

1. Fork repozytorium
2. Utwórz branch dla funkcji
3. Commit zmian
4. Push do branch
5. Utwórz Pull Request

## 📄 Licencja

MIT License - zobacz plik LICENSE dla szczegółów.

## 🔗 Linki

- [LM Studio](https://lmstudio.ai/)
- [Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 📞 Wsparcie

W przypadku problemów:
1. Sprawdź sekcję "Rozwiązywanie Problemów"
2. Przejrzyj logi w katalogu `logs/`
3. Utwórz issue w repozytorium
4. Opisz problem i dołącz logi

---

**Powodzenia w fine-tuningu! 🚀**
