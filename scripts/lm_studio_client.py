#!/usr/bin/env python3
"""
Klient do komunikacji z LM Studio API
"""

import json
import requests
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

class LMStudioClient:
    """Klient do komunikacji z LM Studio"""
    
    def __init__(self, config_path: str = "configs/lm_studio_config.json"):
        """Inicjalizuje klienta LM Studio"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.api_url = self.config['lm_studio']['api_url']
        self.timeout = self.config['lm_studio']['timeout']
        self.max_retries = self.config['lm_studio']['max_retries']
        
        # Sprawdź połączenie
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Sprawdza połączenie z LM Studio"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            if response.status_code == 200:
                logging.info("✅ Połączenie z LM Studio nawiązane")
            else:
                logging.warning(f"⚠️  LM Studio odpowiada z kodem {response.status_code}")
        except Exception as e:
            logging.error(f"❌ Nie można połączyć się z LM Studio: {e}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Pobiera listę dostępnych modeli"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=self.timeout)
            response.raise_for_status()
            return response.json().get('data', [])
        except Exception as e:
            logging.error(f"Błąd podczas pobierania modeli: {e}")
            return []
    
    def generate_completion(self, 
                          prompt: str, 
                          model: str = None,
                          max_tokens: int = None,
                          temperature: float = None,
                          **kwargs) -> Optional[str]:
        """Generuje uzupełnienie tekstu"""
        
        # Użyj wartości domyślnych z konfiguracji
        max_tokens = max_tokens or self.config['lm_studio']['max_tokens']
        temperature = temperature or self.config['lm_studio']['temperature']
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get('top_p', self.config['lm_studio']['top_p']),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config['lm_studio']['frequency_penalty']),
            "presence_penalty": kwargs.get('presence_penalty', self.config['lm_studio']['presence_penalty']),
            "stream": False
        }
        
        # Usuń None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}{self.config['lm_studio']['model_endpoint']}",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['text']
                
            except Exception as e:
                logging.warning(f"Próba {attempt + 1} nieudana: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Wszystkie próby nieudane: {e}")
                    return None
    
    def generate_chat_completion(self,
                               messages: List[Dict[str, str]],
                               model: str = None,
                               max_tokens: int = None,
                               temperature: float = None,
                               **kwargs) -> Optional[str]:
        """Generuje odpowiedź w formacie czatu"""
        
        max_tokens = max_tokens or self.config['lm_studio']['max_tokens']
        temperature = temperature or self.config['lm_studio']['temperature']
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get('top_p', self.config['lm_studio']['top_p']),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config['lm_studio']['frequency_penalty']),
            "presence_penalty": kwargs.get('presence_penalty', self.config['lm_studio']['presence_penalty']),
            "stream": False
        }
        
        payload = {k: v for k, v in payload.items() if v is not None}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}{self.config['lm_studio']['chat_endpoint']}",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except Exception as e:
                logging.warning(f"Próba {attempt + 1} nieudana: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logging.error(f"Wszystkie próby nieudane: {e}")
                    return None
    
    def test_model(self, model_name: str, test_prompt: str = "Cześć! Jak się masz?") -> Dict[str, Any]:
        """Testuje model z przykładowym promptem"""
        logging.info(f"Testowanie modelu: {model_name}")
        
        # Test completion
        start_time = time.time()
        completion = self.generate_completion(test_prompt, model=model_name)
        completion_time = time.time() - start_time
        
        # Test chat
        start_time = time.time()
        chat_response = self.generate_chat_completion(
            [{"role": "user", "content": test_prompt}],
            model=model_name
        )
        chat_time = time.time() - start_time
        
        return {
            "model": model_name,
            "completion": {
                "response": completion,
                "time": completion_time
            },
            "chat": {
                "response": chat_response,
                "time": chat_time
            }
        }
    
    def evaluate_model_on_dataset(self, 
                                model_name: str, 
                                dataset_path: str,
                                output_path: str = None) -> Dict[str, Any]:
        """Ewaluuje model na zbiorze danych"""
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        results = []
        total_time = 0
        
        for i, item in enumerate(dataset):
            if 'instruction' in item:
                prompt = item['instruction']
                expected = item.get('response', '')
            elif 'prompt' in item:
                prompt = item['prompt']
                expected = item.get('completion', '')
            else:
                continue
            
            start_time = time.time()
            response = self.generate_completion(prompt, model=model_name)
            response_time = time.time() - start_time
            total_time += response_time
            
            result = {
                "index": i,
                "prompt": prompt,
                "expected": expected,
                "generated": response,
                "time": response_time
            }
            results.append(result)
            
            if i % 10 == 0:
                logging.info(f"Przetworzono {i}/{len(dataset)} przykładów")
        
        evaluation = {
            "model": model_name,
            "dataset_size": len(dataset),
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(dataset),
            "results": results
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        return evaluation

def main():
    """Przykład użycia klienta LM Studio"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Klient LM Studio")
    parser.add_argument("--action", choices=["test", "list", "evaluate"], 
                       default="list", help="Akcja do wykonania")
    parser.add_argument("--model", help="Nazwa modelu")
    parser.add_argument("--prompt", default="Cześć! Jak się masz?", 
                       help="Prompt do testowania")
    parser.add_argument("--dataset", help="Ścieżka do datasetu do ewaluacji")
    parser.add_argument("--output", help="Ścieżka do zapisania wyników")
    
    args = parser.parse_args()
    
    # Konfiguruj logowanie
    logging.basicConfig(level=logging.INFO)
    
    # Utwórz klienta
    client = LMStudioClient()
    
    if args.action == "list":
        models = client.get_models()
        print("Dostępne modele:")
        for model in models:
            print(f"  - {model.get('id', 'Unknown')}")
    
    elif args.action == "test":
        if not args.model:
            print("Podaj nazwę modelu: --model nazwa")
            return
        
        result = client.test_model(args.model, args.prompt)
        print(f"Model: {result['model']}")
        print(f"Completion: {result['completion']['response']}")
        print(f"Chat: {result['chat']['response']}")
    
    elif args.action == "evaluate":
        if not args.model or not args.dataset:
            print("Podaj model i dataset: --model nazwa --dataset ścieżka")
            return
        
        evaluation = client.evaluate_model_on_dataset(
            args.model, args.dataset, args.output
        )
        print(f"Ewaluacja zakończona: {evaluation['dataset_size']} przykładów")
        print(f"Średni czas na przykład: {evaluation['avg_time_per_sample']:.2f}s")

if __name__ == "__main__":
    main()
