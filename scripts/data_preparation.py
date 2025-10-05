#!/usr/bin/env python3
"""
Narzędzia do przygotowania danych do fine-tuningu
"""

import json
import csv
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import re

def setup_logging():
    """Konfiguruje logowanie"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def clean_text(text: str) -> str:
    """Czyści tekst z niepotrzebnych znaków"""
    # Usuń nadmiarowe białe znaki
    text = re.sub(r'\s+', ' ', text)
    # Usuń znaki specjalne na początku i końcu
    text = text.strip()
    return text

def convert_csv_to_jsonl(input_path: str, output_path: str, 
                        instruction_col: str = "instruction",
                        response_col: str = "response") -> None:
    """Konwertuje plik CSV do formatu JSONL"""
    logging.info(f"Konwertowanie {input_path} do {output_path}")
    
    df = pd.read_csv(input_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            if instruction_col in row and response_col in row:
                data = {
                    "instruction": clean_text(str(row[instruction_col])),
                    "response": clean_text(str(row[response_col]))
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logging.info(f"Konwertowano {len(df)} wierszy")

def convert_txt_to_jsonl(input_path: str, output_path: str, 
                        separator: str = "\n\n") -> None:
    """Konwertuje plik tekstowy do formatu JSONL"""
    logging.info(f"Konwertowanie {input_path} do {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Podziel na sekcje
    sections = content.split(separator)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, section in enumerate(sections):
            if section.strip():
                # Próba automatycznego podziału na instruction/response
                lines = section.strip().split('\n')
                if len(lines) >= 2:
                    instruction = clean_text(lines[0])
                    response = clean_text('\n'.join(lines[1:]))
                    
                    data = {
                        "instruction": instruction,
                        "response": response
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logging.info(f"Utworzono {len([s for s in sections if s.strip()])} przykładów")

def split_dataset(input_path: str, train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    """Dzieli dataset na treningowy, walidacyjny i testowy"""
    logging.info(f"Dzielenie datasetu: {input_path}")
    
    # Wczytaj dane
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Wymieszaj dane
    import random
    random.shuffle(data)
    
    # Oblicz indeksy podziału
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Podziel dane
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Zapisz pliki
    base_path = Path(input_path).parent / Path(input_path).stem
    
    with open(f"{base_path}_train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(f"{base_path}_val.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(f"{base_path}_test.jsonl", 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"Podział: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

def validate_dataset(input_path: str) -> Dict[str, Any]:
    """Waliduje dataset i zwraca statystyki"""
    logging.info(f"Walidacja datasetu: {input_path}")
    
    stats = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "avg_instruction_length": 0,
        "avg_response_length": 0,
        "errors": []
    }
    
    instruction_lengths = []
    response_lengths = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            stats["total_samples"] += 1
            
            try:
                data = json.loads(line.strip())
                
                # Sprawdź wymagane pola
                if "instruction" not in data or "response" not in data:
                    stats["invalid_samples"] += 1
                    stats["errors"].append(f"Linia {i}: Brak wymaganych pól")
                    continue
                
                instruction = str(data["instruction"]).strip()
                response = str(data["response"]).strip()
                
                if not instruction or not response:
                    stats["invalid_samples"] += 1
                    stats["errors"].append(f"Linia {i}: Puste instruction lub response")
                    continue
                
                stats["valid_samples"] += 1
                instruction_lengths.append(len(instruction))
                response_lengths.append(len(response))
                
            except json.JSONDecodeError:
                stats["invalid_samples"] += 1
                stats["errors"].append(f"Linia {i}: Nieprawidłowy JSON")
            except Exception as e:
                stats["invalid_samples"] += 1
                stats["errors"].append(f"Linia {i}: {str(e)}")
    
    # Oblicz statystyki długości
    if instruction_lengths:
        stats["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
    if response_lengths:
        stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)
    
    return stats

def filter_dataset(input_path: str, output_path: str,
                  min_instruction_length: int = 10,
                  max_instruction_length: int = 1000,
                  min_response_length: int = 10,
                  max_response_length: int = 2000) -> None:
    """Filtruje dataset według kryteriów długości"""
    logging.info(f"Filtrowanie datasetu: {input_path}")
    
    filtered_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_count += 1
            
            try:
                data = json.loads(line.strip())
                
                instruction = str(data.get("instruction", "")).strip()
                response = str(data.get("response", "")).strip()
                
                # Sprawdź kryteria długości
                if (min_instruction_length <= len(instruction) <= max_instruction_length and
                    min_response_length <= len(response) <= max_response_length):
                    
                    outfile.write(line)
                    filtered_count += 1
                    
            except json.JSONDecodeError:
                continue
    
    logging.info(f"Przefiltrowano: {filtered_count}/{total_count} próbek")

def create_sample_dataset(output_path: str, num_samples: int = 100) -> None:
    """Tworzy przykładowy dataset do testów"""
    logging.info(f"Tworzenie przykładowego datasetu: {num_samples} próbek")
    
    sample_data = [
        {
            "instruction": "Wyjaśnij czym jest sztuczna inteligencja",
            "response": "Sztuczna inteligencja (AI) to dziedzina informatyki zajmująca się tworzeniem systemów, które mogą wykonywać zadania wymagające ludzkiej inteligencji, takie jak rozpoznawanie mowy, podejmowanie decyzji czy tłumaczenie języków."
        },
        {
            "instruction": "Jak gotować makaron?",
            "response": "1. Zagotuj wodę w dużym garnku. 2. Dodaj sól do wody. 3. Wrzuć makaron i gotuj zgodnie z instrukcją na opakowaniu, mieszając od czasu do czasu. 4. Odcedź makaron i podawaj z ulubionym sosem."
        },
        {
            "instruction": "Co to jest Python?",
            "response": "Python to wysokopoziomowy język programowania ogólnego przeznaczenia, który charakteryzuje się czytelną składnią i wszechstronnością. Jest używany w web development, data science, AI, automatyzacji i wielu innych dziedzinach."
        },
        {
            "instruction": "Jak dbać o rośliny doniczkowe?",
            "response": "1. Podlewaj regularnie, ale nie przelewaj. 2. Zapewnij odpowiednie oświetlenie. 3. Nawoź w sezonie wegetacyjnym. 4. Przesadzaj gdy korzenie wypełnią doniczkę. 5. Usuwaj martwe liście i kwiaty."
        },
        {
            "instruction": "Wyjaśnij czym jest blockchain",
            "response": "Blockchain to rozproszona baza danych, która przechowuje informacje w blokach połączonych kryptograficznie. Każdy blok zawiera dane, hash poprzedniego bloka i timestamp. Technologia zapewnia niezmienność i transparentność danych."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Wybierz losowy przykład i dodaj wariacje
            base = sample_data[i % len(sample_data)].copy()
            base["instruction"] = f"Przykład {i+1}: {base['instruction']}"
            f.write(json.dumps(base, ensure_ascii=False) + '\n')
    
    logging.info(f"Utworzono przykładowy dataset z {num_samples} próbkami")

def main():
    """Główna funkcja skryptu"""
    parser = argparse.ArgumentParser(description="Narzędzia do przygotowania danych")
    parser.add_argument("action", choices=[
        "convert-csv", "convert-txt", "split", "validate", 
        "filter", "create-sample"
    ], help="Akcja do wykonania")
    parser.add_argument("--input", help="Ścieżka do pliku wejściowego")
    parser.add_argument("--output", help="Ścieżka do pliku wyjściowego")
    parser.add_argument("--instruction-col", default="instruction", 
                       help="Nazwa kolumny z instrukcjami (dla CSV)")
    parser.add_argument("--response-col", default="response", 
                       help="Nazwa kolumny z odpowiedziami (dla CSV)")
    parser.add_argument("--separator", default="\\n\\n", 
                       help="Separator sekcji (dla TXT)")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                       help="Proporcja danych treningowych")
    parser.add_argument("--val-ratio", type=float, default=0.1, 
                       help="Proporcja danych walidacyjnych")
    parser.add_argument("--test-ratio", type=float, default=0.1, 
                       help="Proporcja danych testowych")
    parser.add_argument("--min-instruction", type=int, default=10, 
                       help="Minimalna długość instrukcji")
    parser.add_argument("--max-instruction", type=int, default=1000, 
                       help="Maksymalna długość instrukcji")
    parser.add_argument("--min-response", type=int, default=10, 
                       help="Minimalna długość odpowiedzi")
    parser.add_argument("--max-response", type=int, default=2000, 
                       help="Maksymalna długość odpowiedzi")
    parser.add_argument("--num-samples", type=int, default=100, 
                       help="Liczba próbek w przykładowym datasecie")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.action == "convert-csv":
        if not args.input or not args.output:
            print("Podaj --input i --output")
            return
        convert_csv_to_jsonl(args.input, args.output, 
                           args.instruction_col, args.response_col)
    
    elif args.action == "convert-txt":
        if not args.input or not args.output:
            print("Podaj --input i --output")
            return
        convert_txt_to_jsonl(args.input, args.output, args.separator)
    
    elif args.action == "split":
        if not args.input:
            print("Podaj --input")
            return
        split_dataset(args.input, args.train_ratio, args.val_ratio, args.test_ratio)
    
    elif args.action == "validate":
        if not args.input:
            print("Podaj --input")
            return
        stats = validate_dataset(args.input)
        print(f"Statystyki datasetu:")
        print(f"  Łącznie próbek: {stats['total_samples']}")
        print(f"  Prawidłowe: {stats['valid_samples']}")
        print(f"  Nieprawidłowe: {stats['invalid_samples']}")
        print(f"  Średnia długość instrukcji: {stats['avg_instruction_length']:.1f}")
        print(f"  Średnia długość odpowiedzi: {stats['avg_response_length']:.1f}")
        
        if stats['errors']:
            print(f"  Błędy:")
            for error in stats['errors'][:10]:  # Pokaż pierwsze 10 błędów
                print(f"    {error}")
    
    elif args.action == "filter":
        if not args.input or not args.output:
            print("Podaj --input i --output")
            return
        filter_dataset(args.input, args.output,
                      args.min_instruction, args.max_instruction,
                      args.min_response, args.max_response)
    
    elif args.action == "create-sample":
        if not args.output:
            print("Podaj --output")
            return
        create_sample_dataset(args.output, args.num_samples)

if __name__ == "__main__":
    main()
