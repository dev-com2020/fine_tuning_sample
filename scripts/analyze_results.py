#!/usr/bin/env python3
"""
Skrypt do analizy wyników treningu i porównania z innymi modelami
"""

import json
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

def analyze_training_effectiveness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analizuje efektywność treningu"""
    analysis = {
        'convergence_rate': 0.0,
        'overfitting_risk': 'unknown',
        'training_efficiency': 'unknown',
        'recommendations': []
    }
    
    # Analiza zbieżności
    if metrics['loss_history']:
        losses = [entry['loss'] for entry in metrics['loss_history']]
        if len(losses) >= 2:
            initial_loss = losses[0]
            final_loss = losses[-1]
            convergence_rate = (initial_loss - final_loss) / initial_loss
            analysis['convergence_rate'] = convergence_rate
    
    # Analiza overfitting
    if (metrics['final_train_loss'] > 0 and 
        metrics['best_eval_loss'] != float('inf') and 
        metrics['best_eval_loss'] > 0):
        
        ratio = metrics['final_train_loss'] / metrics['best_eval_loss']
        
        if ratio < 0.7:
            analysis['overfitting_risk'] = 'high'
            analysis['recommendations'].append("Wysokie ryzyko underfitting - zwiększ learning rate lub dodaj więcej warstw")
        elif ratio > 1.5:
            analysis['overfitting_risk'] = 'high'
            analysis['recommendations'].append("Wysokie ryzyko overfitting - dodaj regularizację lub zmniejsz learning rate")
        else:
            analysis['overfitting_risk'] = 'low'
            analysis['recommendations'].append("Dobra równowaga między train i eval loss")
    
    # Analiza efektywności
    if metrics['total_training_time'] > 0 and metrics['total_steps'] > 0:
        steps_per_second = metrics['total_steps'] / metrics['total_training_time']
        
        if steps_per_second > 0.01:
            analysis['training_efficiency'] = 'good'
        elif steps_per_second > 0.001:
            analysis['training_efficiency'] = 'moderate'
            analysis['recommendations'].append("Trening może być wolniejszy - rozważ optymalizację")
        else:
            analysis['training_efficiency'] = 'poor'
            analysis['recommendations'].append("Trening bardzo wolny - sprawdź konfigurację GPU/CPU")
    
    # Dodatkowe rekomendacje
    if metrics['final_train_loss'] > 5.0:
        analysis['recommendations'].append("Wysoki final loss - rozważ więcej epok lub zmianę learning rate")
    
    if metrics['total_epochs'] < 2:
        analysis['recommendations'].append("Tylko jedna epoka - rozważ więcej epok dla lepszego treningu")
    
    if metrics['total_steps'] < 10:
        analysis['recommendations'].append("Mało kroków treningu - rozważ większy dataset lub więcej epok")
    
    return analysis

def calculate_model_metrics(model_path: str) -> Dict[str, Any]:
    """Oblicza metryki modelu (rozmiar, parametry itp.)"""
    metrics = {
        'model_size_mb': 0,
        'total_parameters': 0,
        'trainable_parameters': 0,
        'trainable_percentage': 0
    }
    
    try:
        # Sprawdź rozmiar plików modelu
        model_path = Path(model_path)
        if model_path.exists():
            total_size = 0
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            metrics['model_size_mb'] = total_size / (1024 * 1024)
        
        # Wczytaj konfigurację adaptera jeśli istnieje
        adapter_config_path = model_path / 'adapter_config.json'
        if adapter_config_path.exists():
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
                
                # Oblicz parametry LoRA
                if 'r' in adapter_config and 'target_modules' in adapter_config:
                    r = adapter_config['r']
                    target_modules = adapter_config['target_modules']
                    
                    # Przybliżone obliczenie parametrów LoRA
                    # Dla każdego target module: input_dim * r + r * output_dim
                    estimated_params = len(target_modules) * r * 1000  # Przybliżenie
                    metrics['trainable_parameters'] = estimated_params
                    metrics['total_parameters'] = 126799104  # DialoGPT-small
                    metrics['trainable_percentage'] = (estimated_params / metrics['total_parameters']) * 100
    
    except Exception as e:
        logging.error(f"Błąd podczas obliczania metryk modelu: {e}")
    
    return metrics

def create_comprehensive_analysis(metrics: Dict[str, Any], analysis: Dict[str, Any], 
                                model_metrics: Dict[str, Any], output_path: str):
    """Tworzy kompleksową analizę"""
    
    report = f"""
# Kompleksowa Analiza Treningu Modelu

## Podsumowanie Wykonawcze
- **Status**: {'✅ Sukces' if metrics['final_train_loss'] < 10 else '⚠️ Wymaga optymalizacji'}
- **Efektywność**: {analysis['training_efficiency'].title()}
- **Ryzyko Overfitting**: {analysis['overfitting_risk'].title()}
- **Tempo zbieżności**: {analysis['convergence_rate']:.2%}

## Szczegółowe Metryki

### Metryki Treningu
- **Łączne kroki**: {metrics['total_steps']}
- **Epoki**: {metrics['total_epochs']}
- **Czas treningu**: {metrics['total_training_time']:.2f} sekund
- **Final Train Loss**: {metrics['final_train_loss']:.6f}
- **Best Eval Loss**: {metrics['best_eval_loss']:.6f}
- **FLOPS**: {metrics.get('total_flos', 0):,}

### Metryki Modelu
- **Rozmiar modelu**: {model_metrics['model_size_mb']:.2f} MB
- **Łączne parametry**: {model_metrics['total_parameters']:,}
- **Parametry do trenowania**: {model_metrics['trainable_parameters']:,}
- **Procent trainable**: {model_metrics['trainable_percentage']:.2f}%

### Analiza Efektywności
- **Kroki/sekundę**: {metrics['total_steps'] / max(metrics['total_training_time'], 1):.4f}
- **Loss na krok**: {metrics['final_train_loss'] / max(metrics['total_steps'], 1):.6f}
- **Czas na epokę**: {metrics['total_training_time'] / max(metrics['total_epochs'], 1):.2f}s

## Historia Learning Rate
"""
    
    if metrics['learning_rate_schedule']:
        for entry in metrics['learning_rate_schedule']:
            report += f"- Krok {entry['step']}: {entry['lr']:.2e}\n"
    else:
        report += "- Brak danych o learning rate\n"
    
    # Historia Loss
    report += "\n## Historia Loss\n"
    if metrics['loss_history']:
        report += "### Train Loss\n"
        for entry in metrics['loss_history'][:5]:  # Pierwsze 5
            report += f"- Krok {entry['step']}: {entry['loss']:.6f}\n"
        
        if len(metrics['loss_history']) > 5:
            report += "...\n"
            for entry in metrics['loss_history'][-3:]:  # Ostatnie 3
                report += f"- Krok {entry['step']}: {entry['loss']:.6f}\n"
    else:
        report += "- Brak historii train loss\n"
    
    if metrics['eval_loss_history']:
        report += "\n### Eval Loss\n"
        for entry in metrics['eval_loss_history'][:5]:  # Pierwsze 5
            report += f"- Krok {entry['step']}: {entry['loss']:.6f}\n"
    else:
        report += "- Brak historii eval loss\n"
    
    # Rekomendacje
    report += "\n## Rekomendacje i Wnioski\n"
    
    if analysis['recommendations']:
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"{i}. {rec}\n"
    else:
        report += "- Brak konkretnych rekomendacji - trening przebiegł poprawnie\n"
    
    # Dodatkowe analizy
    report += "\n## Dodatkowe Analizy\n"
    
    # Analiza rozmiaru modelu
    if model_metrics['model_size_mb'] > 1000:
        report += "- **Rozmiar modelu**: Duży (>1GB) - rozważ kompresję\n"
    elif model_metrics['model_size_mb'] > 100:
        report += "- **Rozmiar modelu**: Umiarkowany (100MB-1GB)\n"
    else:
        report += "- **Rozmiar modelu**: Mały (<100MB) - dobry do testów\n"
    
    # Analiza parametrów
    if model_metrics['trainable_percentage'] > 10:
        report += "- **Parametry**: Wysoki procent trainable - możliwe overfitting\n"
    elif model_metrics['trainable_percentage'] > 1:
        report += "- **Parametry**: Optymalny procent trainable (1-10%)\n"
    else:
        report += "- **Parametry**: Niski procent trainable - możliwe underfitting\n"
    
    # Analiza czasu treningu
    if metrics['total_training_time'] > 3600:  # 1 godzina
        report += "- **Czas treningu**: Długi (>1h) - rozważ optymalizację\n"
    elif metrics['total_training_time'] > 300:  # 5 minut
        report += "- **Czas treningu**: Umiarkowany (5min-1h)\n"
    else:
        report += "- **Czas treningu**: Krótki (<5min) - dobry do eksperymentów\n"
    
    # Zapisz raport
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info(f"Kompleksowa analiza zapisana: {output_path}")

def create_comparison_visualization(metrics: Dict[str, Any], model_metrics: Dict[str, Any], 
                                  output_path: str):
    """Tworzy wizualizację porównawczą"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Kompleksowa Analiza Treningu Modelu', fontsize=16, fontweight='bold')
    
    # Wykres 1: Metryki treningu
    training_metrics = {
        'Kroki': metrics['total_steps'],
        'Epoki': metrics['total_epochs'],
        'Czas (s)': metrics['total_training_time'],
        'Final Loss': metrics['final_train_loss']
    }
    
    axes[0, 0].bar(training_metrics.keys(), training_metrics.values())
    axes[0, 0].set_title('Metryki Treningu')
    axes[0, 0].set_ylabel('Wartość')
    
    # Wykres 2: Metryki modelu
    model_metrics_plot = {
        'Rozmiar (MB)': model_metrics['model_size_mb'],
        'Parametry (M)': model_metrics['total_parameters'] / 1e6,
        'Trainable (M)': model_metrics['trainable_parameters'] / 1e6,
        'Trainable (%)': model_metrics['trainable_percentage']
    }
    
    bars = axes[0, 1].bar(model_metrics_plot.keys(), model_metrics_plot.values())
    axes[0, 1].set_title('Metryki Modelu')
    axes[0, 1].set_ylabel('Wartość')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, model_metrics_plot.values()):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    # Wykres 3: Historia learning rate
    if metrics['learning_rate_schedule']:
        lr_data = pd.DataFrame(metrics['learning_rate_schedule'])
        axes[0, 2].plot(lr_data['step'], lr_data['lr'], marker='o')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_xlabel('Krok')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
    else:
        axes[0, 2].text(0.5, 0.5, 'Brak danych\nLearning Rate', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Learning Rate Schedule')
    
    # Wykres 4: Historia loss
    if metrics['loss_history']:
        loss_data = pd.DataFrame(metrics['loss_history'])
        axes[1, 0].plot(loss_data['step'], loss_data['loss'], marker='o', label='Train Loss')
    else:
        axes[1, 0].text(0.5, 0.5, 'Brak danych\nTrain Loss', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    if metrics['eval_loss_history']:
        eval_data = pd.DataFrame(metrics['eval_loss_history'])
        axes[1, 0].plot(eval_data['step'], eval_data['loss'], marker='s', label='Eval Loss')
    
    axes[1, 0].set_title('Historia Loss')
    axes[1, 0].set_xlabel('Krok')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Wykres 5: Efektywność treningu
    efficiency_metrics = {
        'Kroki/s': metrics['total_steps'] / max(metrics['total_training_time'], 1),
        'Loss/krok': metrics['final_train_loss'] / max(metrics['total_steps'], 1),
        'Czas/epoka': metrics['total_training_time'] / max(metrics['total_epochs'], 1)
    }
    
    bars = axes[1, 1].bar(efficiency_metrics.keys(), efficiency_metrics.values())
    axes[1, 1].set_title('Efektywność Treningu')
    axes[1, 1].set_ylabel('Wartość')
    
    # Wykres 6: Podsumowanie jakości
    quality_scores = {
        'Zbieżność': min(1.0, max(0.0, (10 - metrics['final_train_loss']) / 10)),
        'Efektywność': min(1.0, max(0.0, (metrics['total_steps'] / max(metrics['total_training_time'], 1)) * 100)),
        'Rozmiar': min(1.0, max(0.0, (100 - model_metrics['model_size_mb']) / 100)),
        'Trainable %': min(1.0, max(0.0, model_metrics['trainable_percentage'] / 10))
    }
    
    bars = axes[1, 2].bar(quality_scores.keys(), quality_scores.values())
    axes[1, 2].set_title('Wskaźniki Jakości (0-1)')
    axes[1, 2].set_ylabel('Wynik')
    axes[1, 2].set_ylim(0, 1)
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, quality_scores.values()):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Wizualizacja porównawcza zapisana: {output_path}")

def main():
    """Główna funkcja skryptu"""
    parser = argparse.ArgumentParser(description="Analiza wyników treningu")
    parser.add_argument("--metrics-file", default="./output/metrics/training_metrics.json", 
                       help="Plik z metrykami treningu")
    parser.add_argument("--model-dir", default="./output", 
                       help="Katalog z modelem")
    parser.add_argument("--output-dir", default="./output/analysis", 
                       help="Katalog wyjściowy")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Wczytaj metryki
    with open(args.metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    # Oblicz metryki modelu
    model_metrics = calculate_model_metrics(args.model_dir)
    
    # Analizuj efektywność
    analysis = analyze_training_effectiveness(metrics)
    
    # Utwórz katalog wyjściowy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wyświetl podstawowe wyniki
    print("\n" + "="*60)
    print("ANALIZA WYNIKÓW TRENINGU")
    print("="*60)
    print(f"Efektywność treningu: {analysis['training_efficiency'].title()}")
    print(f"Ryzyko overfitting: {analysis['overfitting_risk'].title()}")
    print(f"Tempo zbieżności: {analysis['convergence_rate']:.2%}")
    print(f"Rozmiar modelu: {model_metrics['model_size_mb']:.2f} MB")
    print(f"Parametry trainable: {model_metrics['trainable_percentage']:.2f}%")
    print("\nRekomendacje:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"{i}. {rec}")
    print("="*60)
    
    # Utwórz kompleksową analizę
    analysis_report_path = output_dir / "comprehensive_analysis.md"
    create_comprehensive_analysis(metrics, analysis, model_metrics, str(analysis_report_path))
    
    # Utwórz wizualizację
    visualization_path = output_dir / "comprehensive_analysis.png"
    create_comparison_visualization(metrics, model_metrics, str(visualization_path))
    
    # Zapisz analizę jako JSON
    analysis_data = {
        'metrics': metrics,
        'model_metrics': model_metrics,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    analysis_json_path = output_dir / "analysis_data.json"
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    logging.info("Analiza zakończona pomyślnie!")
    print(f"\nWyniki zapisane w: {output_dir}")

if __name__ == "__main__":
    main()
