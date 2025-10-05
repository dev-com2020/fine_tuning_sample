#!/usr/bin/env python3
"""
Skrypt do tworzenia metryk i raportów z treningu
"""

import json
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Konfiguruje logowanie"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_training_state(training_args_path: str) -> Dict[str, Any]:
    """Wczytuje stan treningu z pliku training_args.bin"""
    try:
        import torch
        state = torch.load(training_args_path, map_location='cpu', weights_only=False)
        return state
    except Exception as e:
        logging.error(f"Nie można wczytać stanu treningu: {e}")
        return {}

def load_trainer_state(trainer_state_path: str) -> Dict[str, Any]:
    """Wczytuje stan trainera z pliku trainer_state.json"""
    try:
        with open(trainer_state_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Nie można wczytać stanu trainera: {e}")
        return {}

def parse_tensorboard_logs(logs_dir: str) -> pd.DataFrame:
    """Parsuje logi TensorBoard"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Znajdź pliki event
        event_files = list(Path(logs_dir).glob("**/events.out.tfevents.*"))
        
        if not event_files:
            logging.warning("Nie znaleziono plików TensorBoard")
            return pd.DataFrame()
        
        # Wczytaj dane z pierwszego pliku
        ea = EventAccumulator(str(event_files[0]))
        ea.Reload()
        
        data = []
        
        # Pobierz dostępne tagi
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            for event in scalar_events:
                data.append({
                    'step': event.step,
                    'value': event.value,
                    'wall_time': event.wall_time,
                    'tag': tag
                })
        
        return pd.DataFrame(data)
        
    except ImportError:
        logging.warning("TensorBoard nie jest zainstalowany")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Błąd podczas parsowania logów TensorBoard: {e}")
        return pd.DataFrame()

def calculate_training_metrics(trainer_state: Dict[str, Any]) -> Dict[str, Any]:
    """Oblicza metryki treningu"""
    metrics = {
        'total_steps': 0,
        'total_epochs': 0,
        'total_training_time': 0,
        'final_train_loss': 0,
        'best_eval_loss': float('inf'),
        'learning_rate_schedule': [],
        'loss_history': [],
        'eval_loss_history': []
    }
    
    if not trainer_state:
        return metrics
    
    # Podstawowe informacje
    if 'global_step' in trainer_state:
        metrics['total_steps'] = trainer_state['global_step']
    
    if 'epoch' in trainer_state:
        metrics['total_epochs'] = trainer_state['epoch']
    
    if 'total_flos' in trainer_state:
        metrics['total_flos'] = trainer_state['total_flos']
    
    # Historia treningu
    if 'log_history' in trainer_state:
        log_history = trainer_state['log_history']
        
        for entry in log_history:
            if 'train_loss' in entry:
                metrics['loss_history'].append({
                    'step': entry.get('step', 0),
                    'loss': entry['train_loss']
                })
            
            if 'eval_loss' in entry:
                metrics['eval_loss_history'].append({
                    'step': entry.get('step', 0),
                    'loss': entry['eval_loss']
                })
                
                if entry['eval_loss'] < metrics['best_eval_loss']:
                    metrics['best_eval_loss'] = entry['eval_loss']
            
            if 'learning_rate' in entry:
                metrics['learning_rate_schedule'].append({
                    'step': entry.get('step', 0),
                    'lr': entry['learning_rate']
                })
        
        # Finalne wartości
        if metrics['loss_history']:
            metrics['final_train_loss'] = metrics['loss_history'][-1]['loss']
    
    # Czas treningu
    if 'train_runtime' in trainer_state:
        metrics['total_training_time'] = trainer_state['train_runtime']
    
    return metrics

def create_training_visualizations(metrics: Dict[str, Any], tensorboard_data: pd.DataFrame, output_dir: str):
    """Tworzy wizualizacje treningu"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ustawienia stylu
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Wykres 1: Historia loss
    if metrics['loss_history'] or metrics['eval_loss_history']:
        plt.figure(figsize=(12, 8))
        
        if metrics['loss_history']:
            loss_df = pd.DataFrame(metrics['loss_history'])
            plt.plot(loss_df['step'], loss_df['loss'], label='Train Loss', linewidth=2)
        
        if metrics['eval_loss_history']:
            eval_df = pd.DataFrame(metrics['eval_loss_history'])
            plt.plot(eval_df['step'], eval_df['loss'], label='Eval Loss', linewidth=2)
        
        plt.xlabel('Kroki treningu')
        plt.ylabel('Loss')
        plt.title('Historia Loss podczas treningu')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Wykres 2: Learning rate
    if metrics['learning_rate_schedule']:
        plt.figure(figsize=(12, 6))
        lr_df = pd.DataFrame(metrics['learning_rate_schedule'])
        plt.plot(lr_df['step'], lr_df['lr'], linewidth=2, color='green')
        plt.xlabel('Kroki treningu')
        plt.ylabel('Learning Rate')
        plt.title('Harmonogram Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Wykres 3: TensorBoard data
    if not tensorboard_data.empty:
        plt.figure(figsize=(15, 10))
        
        # Podziel na subploty dla różnych tagów
        unique_tags = tensorboard_data['tag'].unique()
        n_tags = len(unique_tags)
        
        if n_tags > 0:
            cols = min(2, n_tags)
            rows = (n_tags + cols - 1) // cols
            
            for i, tag in enumerate(unique_tags):
                plt.subplot(rows, cols, i + 1)
                tag_data = tensorboard_data[tensorboard_data['tag'] == tag]
                plt.plot(tag_data['step'], tag_data['value'], linewidth=2)
                plt.title(f'TensorBoard: {tag}')
                plt.xlabel('Kroki')
                plt.ylabel('Wartość')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'tensorboard_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Wykres 4: Podsumowanie metryk
    plt.figure(figsize=(12, 8))
    
    # Przygotuj dane do wykresu słupkowego
    metric_names = []
    metric_values = []
    
    if metrics['total_steps'] > 0:
        metric_names.append('Łączne kroki')
        metric_values.append(metrics['total_steps'])
    
    if metrics['total_epochs'] > 0:
        metric_names.append('Epoki')
        metric_values.append(metrics['total_epochs'])
    
    if metrics['total_training_time'] > 0:
        metric_names.append('Czas treningu (s)')
        metric_values.append(metrics['total_training_time'])
    
    if metrics['final_train_loss'] > 0:
        metric_names.append('Final Train Loss')
        metric_values.append(metrics['final_train_loss'])
    
    if metrics['best_eval_loss'] != float('inf'):
        metric_names.append('Best Eval Loss')
        metric_values.append(metrics['best_eval_loss'])
    
    if metric_names:
        bars = plt.bar(metric_names, metric_values)
        plt.title('Podsumowanie metryk treningu')
        plt.ylabel('Wartość')
        
        # Dodaj wartości na słupkach
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'training_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Wizualizacje zapisane w katalogu: {output_path}")

def create_training_report(metrics: Dict[str, Any], config: Dict[str, Any], output_path: str):
    """Tworzy raport treningu"""
    
    report = f"""
# Raport Treningu Modelu

## Informacje Podstawowe
- **Data treningu**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model**: {config.get('model', {}).get('name', 'Nieznany')}
- **Epoki**: {metrics['total_epochs']}
- **Łączne kroki**: {metrics['total_steps']}
- **Czas treningu**: {metrics['total_training_time']:.2f} sekund

## Metryki Treningu
- **Final Train Loss**: {metrics['final_train_loss']:.6f}
- **Best Eval Loss**: {metrics['best_eval_loss']:.6f}
- **FLOPS**: {metrics.get('total_flos', 'Nieznane')}

## Konfiguracja Treningu
- **Learning Rate**: {config.get('training', {}).get('learning_rate', 'Nieznane')}
- **Batch Size**: {config.get('training', {}).get('per_device_train_batch_size', 'Nieznane')}
- **Gradient Accumulation Steps**: {config.get('training', {}).get('gradient_accumulation_steps', 'Nieznane')}
- **Max Length**: {config.get('model', {}).get('max_length', 'Nieznane')}

## LoRA Konfiguracja
- **Włączone**: {config.get('lora', {}).get('enabled', False)}
- **Rank (r)**: {config.get('lora', {}).get('r', 'Nieznane')}
- **Alpha**: {config.get('lora', {}).get('lora_alpha', 'Nieznane')}
- **Dropout**: {config.get('lora', {}).get('lora_dropout', 'Nieznane')}

## Historia Treningu
"""
    
    # Dodaj historię loss
    if metrics['loss_history']:
        report += "\n### Historia Train Loss\n"
        for entry in metrics['loss_history'][-5:]:  # Ostatnie 5 wpisów
            report += f"- Krok {entry['step']}: {entry['loss']:.6f}\n"
    
    if metrics['eval_loss_history']:
        report += "\n### Historia Eval Loss\n"
        for entry in metrics['eval_loss_history'][-5:]:  # Ostatnie 5 wpisów
            report += f"- Krok {entry['step']}: {entry['loss']:.6f}\n"
    
    # Dodaj wnioski
    report += "\n## Wnioski\n"
    
    if metrics['best_eval_loss'] != float('inf') and metrics['final_train_loss'] > 0:
        overfitting_ratio = metrics['final_train_loss'] / metrics['best_eval_loss']
        if overfitting_ratio > 1.5:
            report += "- **Uwaga**: Możliwe overfitting (train loss znacznie niższy niż eval loss)\n"
        elif overfitting_ratio < 0.7:
            report += "- **Uwaga**: Możliwe underfitting (train loss wyższy niż eval loss)\n"
        else:
            report += "- **Dobrze**: Train i eval loss są w równowadze\n"
    
    if metrics['total_training_time'] > 0:
        steps_per_second = metrics['total_steps'] / metrics['total_training_time']
        report += f"- **Wydajność**: {steps_per_second:.4f} kroków/sekundę\n"
    
    if metrics['final_train_loss'] < 1.0:
        report += "- **Wynik**: Niski final loss - model dobrze się uczył\n"
    elif metrics['final_train_loss'] < 5.0:
        report += "- **Wynik**: Umiarkowany final loss - możliwa potrzeba więcej epok\n"
    else:
        report += "- **Wynik**: Wysoki final loss - rozważ zmianę learning rate lub architektury\n"
    
    # Zapisz raport
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info(f"Raport zapisany: {output_path}")

def main():
    """Główna funkcja skryptu"""
    parser = argparse.ArgumentParser(description="Tworzenie metryk i raportów z treningu")
    parser.add_argument("--output-dir", default="./output", help="Katalog z wynikami treningu")
    parser.add_argument("--config", default="configs/training_config.yaml", help="Plik konfiguracyjny")
    parser.add_argument("--metrics-output", default="./output/metrics", help="Katalog wyjściowy dla metryk")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Wczytaj konfigurację
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_path = Path(args.output_dir)
    metrics_output = Path(args.metrics_output)
    metrics_output.mkdir(parents=True, exist_ok=True)
    
    logging.info("Tworzenie metryk z treningu...")
    
    # Wczytaj stan treningu
    trainer_state_path = output_path / "checkpoint-2" / "trainer_state.json"
    training_args_path = output_path / "training_args.bin"
    
    trainer_state = load_trainer_state(str(trainer_state_path))
    training_state = load_training_state(str(training_args_path))
    
    # Parsuj logi TensorBoard
    tensorboard_logs_dir = output_path / "runs"
    tensorboard_data = parse_tensorboard_logs(str(tensorboard_logs_dir))
    
    # Oblicz metryki
    metrics = calculate_training_metrics(trainer_state)
    
    # Wyświetl podstawowe metryki
    print("\n" + "="*50)
    print("METRYKI TRENINGU")
    print("="*50)
    print(f"Łączne kroki: {metrics['total_steps']}")
    print(f"Epoki: {metrics['total_epochs']}")
    print(f"Czas treningu: {metrics['total_training_time']:.2f}s")
    print(f"Final Train Loss: {metrics['final_train_loss']:.6f}")
    print(f"Best Eval Loss: {metrics['best_eval_loss']:.6f}")
    print(f"Liczba punktów loss: {len(metrics['loss_history'])}")
    print(f"Liczba punktów eval: {len(metrics['eval_loss_history'])}")
    print("="*50)
    
    # Utwórz wizualizacje
    create_training_visualizations(metrics, tensorboard_data, str(metrics_output))
    
    # Utwórz raport
    report_path = metrics_output / "training_report.md"
    create_training_report(metrics, config, str(report_path))
    
    # Zapisz metryki jako JSON
    metrics_path = metrics_output / "training_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logging.info("Metryki utworzone pomyślnie!")
    print(f"\nWyniki zapisane w: {metrics_output}")

if __name__ == "__main__":
    main()
