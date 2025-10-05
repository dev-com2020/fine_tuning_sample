#!/usr/bin/env python3
"""
Skrypt do konfiguracji środowiska fine-tuningu
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path

def check_python_version():
    """Sprawdza wersję Pythona"""
    if sys.version_info < (3, 8):
        print("Wymagany Python 3.8 lub nowszy")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")

def install_requirements():
    """Instaluje wymagane pakiety"""
    print("Instalowanie wymaganych pakietów...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("Pakiety zainstalowane pomyślnie")
    except subprocess.CalledProcessError as e:
        print(f"Błąd instalacji pakietów: {e}")
        sys.exit(1)

def create_directories():
    """Tworzy potrzebne katalogi"""
    directories = [
        "data/raw",
        "data/processed", 
        "models/base",
        "models/fine_tuned",
        "output/checkpoints",
        "output/logs",
        "logs/training",
        "logs/evaluation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Utworzono katalog: {directory}")

def check_cuda():
    """Sprawdza dostępność CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA dostępne - {torch.cuda.get_device_name(0)}")
            print(f"   Pamięć GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA niedostępne - będzie używany CPU")
    except ImportError:
            print("PyTorch nie zainstalowany")

def check_lm_studio_connection():
    """Sprawdza połączenie z LM Studio"""
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("LM Studio działa")
            print(f"   Dostępne modele: {len(models.get('data', []))}")
        else:
            print("LM Studio nie odpowiada na porcie 1234")
    except Exception as e:
        print(f"Nie można połączyć się z LM Studio: {e}")

def create_config_template():
    """Tworzy szablon konfiguracji"""
    config_template = {
        "model_name": "twoj-model",
        "base_model_path": "./models/base/",
        "output_path": "./models/fine_tuned/",
        "data_path": "./data/processed/",
        "training_epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": 4
    }
    
    with open("configs/user_config.json", "w", encoding="utf-8") as f:
        json.dump(config_template, f, indent=2, ensure_ascii=False)
    print("Utworzono szablon konfiguracji: configs/user_config.json")

def main():
    print("Konfiguracja środowiska fine-tuningu")
    print("=" * 50)
    
    check_python_version()
    create_directories()
    install_requirements()
    check_cuda()
    check_lm_studio_connection()
    create_config_template()
    
    print("\nŚrodowisko skonfigurowane pomyślnie!")
    print("\nNastępne kroki:")
    print("1. Umieść swoje modele w katalogu models/base/")
    print("2. Przygotuj dane treningowe w formacie JSONL")
    print("3. Skonfiguruj parametry w configs/user_config.json")
    print("4. Uruchom fine-tuning: python scripts/fine_tune.py")

if __name__ == "__main__":
    main()
