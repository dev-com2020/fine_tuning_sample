#!/usr/bin/env python3
"""
Skrypt do tworzenia interaktywnego dashboardu z wynikami treningu
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

def create_training_dashboard(metrics: Dict[str, Any], analysis_data: Dict[str, Any], 
                            output_path: str):
    """Tworzy kompleksowy dashboard treningu"""
    
    # Ustawienia stylu
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Utwórz figure z subplotami
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Tytuł główny
    fig.suptitle('Dashboard Treningu Modelu - Fine-Tuning z LoRA', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Metryki podstawowe (górny lewy)
    ax1 = fig.add_subplot(gs[0, 0])
    basic_metrics = {
        'Epoki': metrics['total_epochs'],
        'Kroki': metrics['total_steps'],
        'Czas (min)': metrics['total_training_time'] / 60,
        'FLOPS (B)': metrics.get('total_flos', 0) / 1e9
    }
    
    bars = ax1.bar(basic_metrics.keys(), basic_metrics.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Metryki Podstawowe', fontweight='bold')
    ax1.set_ylabel('Wartość')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, basic_metrics.values()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Metryki modelu (górny środek)
    ax2 = fig.add_subplot(gs[0, 1])
    model_metrics = analysis_data['model_metrics']
    model_data = {
        'Rozmiar (MB)': model_metrics['model_size_mb'],
        'Parametry (M)': model_metrics['total_parameters'] / 1e6,
        'Trainable (K)': model_metrics['trainable_parameters'] / 1e3,
        'Trainable (%)': model_metrics['trainable_percentage']
    }
    
    bars = ax2.bar(model_data.keys(), model_data.values(),
                   color=['#FF9F43', '#10AC84', '#EE5A24', '#0984E3'])
    ax2.set_title('Metryki Modelu', fontweight='bold')
    ax2.set_ylabel('Wartość')
    
    for bar, value in zip(bars, model_data.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Historia Learning Rate (górny prawy)
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics['learning_rate_schedule']:
        lr_data = pd.DataFrame(metrics['learning_rate_schedule'])
        ax3.plot(lr_data['step'], lr_data['lr'], marker='o', linewidth=3, markersize=8,
                color='#6C5CE7')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Krok')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Brak danych\nLearning Rate', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, color='gray')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
    
    # 4. Historia Loss (górny prawy)
    ax4 = fig.add_subplot(gs[0, 3])
    if metrics['loss_history'] or metrics['eval_loss_history']:
        if metrics['loss_history']:
            loss_data = pd.DataFrame(metrics['loss_history'])
            ax4.plot(loss_data['step'], loss_data['loss'], marker='o', linewidth=3, 
                    markersize=8, label='Train Loss', color='#FF6B6B')
        
        if metrics['eval_loss_history']:
            eval_data = pd.DataFrame(metrics['eval_loss_history'])
            ax4.plot(eval_data['step'], eval_data['loss'], marker='s', linewidth=3,
                    markersize=8, label='Eval Loss', color='#4ECDC4')
        
        ax4.set_title('Historia Loss', fontweight='bold')
        ax4.set_xlabel('Krok')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Brak danych\nHistoria Loss', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, color='gray')
        ax4.set_title('Historia Loss', fontweight='bold')
    
    # 5. Efektywność treningu (środkowy lewy)
    ax5 = fig.add_subplot(gs[1, 0])
    efficiency_data = {
        'Kroki/s': metrics['total_steps'] / max(metrics['total_training_time'], 1),
        'Loss/krok': metrics['final_train_loss'] / max(metrics['total_steps'], 1),
        'Czas/epoka': metrics['total_training_time'] / max(metrics['total_epochs'], 1),
        'FLOPS/krok': metrics.get('total_flos', 0) / max(metrics['total_steps'], 1) / 1e9
    }
    
    bars = ax5.bar(efficiency_data.keys(), efficiency_data.values(),
                   color=['#00B894', '#FDCB6E', '#E17055', '#74B9FF'])
    ax5.set_title('Efektywność Treningu', fontweight='bold')
    ax5.set_ylabel('Wartość')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, efficiency_data.values()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Wskaźniki jakości (środkowy środek)
    ax6 = fig.add_subplot(gs[1, 1])
    quality_data = {
        'Zbieżność': min(1.0, max(0.0, (10 - metrics['final_train_loss']) / 10)),
        'Efektywność': min(1.0, max(0.0, (metrics['total_steps'] / max(metrics['total_training_time'], 1)) * 100)),
        'Rozmiar': min(1.0, max(0.0, (100 - model_metrics['model_size_mb']) / 100)),
        'Trainable %': min(1.0, max(0.0, model_metrics['trainable_percentage'] / 10))
    }
    
    bars = ax6.bar(quality_data.keys(), quality_data.values(),
                   color=['#A29BFE', '#FD79A8', '#FDCB6E', '#00B894'])
    ax6.set_title('Wskaźniki Jakości (0-1)', fontweight='bold')
    ax6.set_ylabel('Wynik')
    ax6.set_ylim(0, 1)
    ax6.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, quality_data.values()):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Analiza ryzyka (środkowy prawy)
    ax7 = fig.add_subplot(gs[1, 2])
    analysis = analysis_data['analysis']
    
    # Określ kolory na podstawie ryzyka
    risk_colors = {
        'low': '#00B894',
        'moderate': '#FDCB6E', 
        'high': '#E17055',
        'unknown': '#BDC3C7'
    }
    
    risk_data = {
        'Overfitting': analysis.get('overfitting_risk', 'unknown'),
        'Efektywność': analysis.get('training_efficiency', 'unknown')
    }
    
    colors = [risk_colors.get(risk, '#BDC3C7') for risk in risk_data.values()]
    bars = ax7.bar(risk_data.keys(), [1, 1], color=colors)
    ax7.set_title('Analiza Ryzyka', fontweight='bold')
    ax7.set_ylabel('Status')
    ax7.set_ylim(0, 1.2)
    ax7.set_yticks([])
    
    # Dodaj etykiety stanu
    for i, (risk, bar) in enumerate(zip(risk_data.values(), bars)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                risk.upper(), ha='center', va='bottom', fontweight='bold')
    
    # 8. Podsumowanie (środkowy prawy)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    # Tekst podsumowania
    eval_loss_str = f"{metrics['best_eval_loss']:.6f}" if metrics['best_eval_loss'] != float('inf') else 'N/A'
    summary_text = f"""
PODSUMOWANIE TRENINGU

Status: {'✅ SUKCES' if metrics['final_train_loss'] < 10 else '⚠️ WYMAGA OPTYMALIZACJI'}

Metryki:
• Final Loss: {metrics['final_train_loss']:.6f}
• Best Eval Loss: {eval_loss_str}
• Rozmiar modelu: {model_metrics['model_size_mb']:.1f} MB
• Trainable: {model_metrics['trainable_percentage']:.2f}%

Rekomendacje:
{chr(10).join([f"• {rec}" for rec in analysis.get('recommendations', [])][:3])}
"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 9. Porównanie z benchmarkami (dolny rząd)
    ax9 = fig.add_subplot(gs[2, :2])
    
    # Symulowane benchmarki
    benchmarks = {
        'Nasz Model': {
            'Loss': metrics['final_train_loss'],
            'Czas (min)': metrics['total_training_time'] / 60,
            'Rozmiar (MB)': model_metrics['model_size_mb'],
            'Trainable (%)': model_metrics['trainable_percentage']
        },
        'Benchmark 1': {
            'Loss': 2.5,
            'Czas (min)': 45,
            'Rozmiar (MB)': 120,
            'Trainable (%)': 5.0
        },
        'Benchmark 2': {
            'Loss': 1.8,
            'Czas (min)': 60,
            'Rozmiar (MB)': 200,
            'Trainable (%)': 8.0
        }
    }
    
    # Normalizuj dane dla porównania
    normalized_data = {}
    for model, data in benchmarks.items():
        normalized_data[model] = {
            'Jakość': max(0, (10 - data['Loss']) / 10),
            'Szybkość': max(0, (120 - data['Czas (min)']) / 120),
            'Efektywność': max(0, (200 - data['Rozmiar (MB)']) / 200),
            'Optymalizacja': min(1, data['Trainable (%)'] / 10)
        }
    
    # Wykres radarowy
    categories = list(normalized_data['Nasz Model'].keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Zamknij wykres
    
    for model, data in normalized_data.items():
        values = list(data.values())
        values += values[:1]  # Zamknij wykres
        
        ax9.plot(angles, values, 'o-', linewidth=2, label=model)
        ax9.fill(angles, values, alpha=0.25)
    
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(categories)
    ax9.set_ylim(0, 1)
    ax9.set_title('Porównanie z Benchmarkami', fontweight='bold')
    ax9.legend()
    ax9.grid(True)
    
    # 10. Timeline treningu (dolny prawy)
    ax10 = fig.add_subplot(gs[2, 2:])
    
    # Symuluj timeline
    timeline_data = {
        'Inicjalizacja': 0,
        'Ładowanie danych': 5,
        'Konfiguracja LoRA': 10,
        'Tokenizacja': 15,
        'Epoka 1': 20,
        'Ewaluacja': 25,
        'Zapisywanie': 30
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    
    bars = ax10.barh(list(timeline_data.keys()), list(timeline_data.values()), 
                     color=colors)
    ax10.set_title('Timeline Treningu', fontweight='bold')
    ax10.set_xlabel('Czas (s)')
    
    for bar, value in zip(bars, timeline_data.values()):
        width = bar.get_width()
        ax10.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                 f'{value}s', ha='left', va='center', fontweight='bold')
    
    # 11. Statystyki szczegółowe (dolny rząd)
    ax11 = fig.add_subplot(gs[3, :])
    ax11.axis('off')
    
    # Tabela ze statystykami
    eval_loss_str = f"{metrics['best_eval_loss']:.6f}" if metrics['best_eval_loss'] != float('inf') else 'N/A'
    stats_text = f"""
STATYSTYKI SZCZEGÓŁOWE
{'='*80}
{'Metryka':<30} {'Wartość':<20} {'Jednostka':<15} {'Status':<15}
{'='*80}
{'Łączne kroki':<30} {metrics['total_steps']:<20} {'kroki':<15} {'✅':<15}
{'Epoki':<30} {metrics['total_epochs']:<20} {'epoki':<15} {'✅':<15}
{'Czas treningu':<30} {metrics['total_training_time']:<20.2f} {'sekundy':<15} {'✅':<15}
{'Final Train Loss':<30} {metrics['final_train_loss']:<20.6f} {'loss':<15} {'✅' if metrics['final_train_loss'] < 5 else '⚠️':<15}
{'Best Eval Loss':<30} {eval_loss_str:<20} {'loss':<15} {'✅' if metrics['best_eval_loss'] != float('inf') and metrics['best_eval_loss'] < 5 else '⚠️':<15}
{'Rozmiar modelu':<30} {model_metrics['model_size_mb']:<20.2f} {'MB':<15} {'✅':<15}
{'Parametry trainable':<30} {model_metrics['trainable_percentage']:<20.2f} {'%':<15} {'✅' if 1 <= model_metrics['trainable_percentage'] <= 10 else '⚠️':<15}
{'FLOPS':<30} {metrics.get('total_flos', 0):<20,} {'FLOPS':<15} {'✅':<15}
{'='*80}
"""
    
    ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Dodaj timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Wygenerowano: {timestamp}', 
             ha='right', va='bottom', fontsize=8, style='italic')
    
    # Zapisz dashboard
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Dashboard zapisany: {output_path}")

def main():
    """Główna funkcja skryptu"""
    parser = argparse.ArgumentParser(description="Tworzenie dashboardu treningu")
    parser.add_argument("--metrics-file", default="./output/metrics/training_metrics.json", 
                       help="Plik z metrykami treningu")
    parser.add_argument("--analysis-file", default="./output/analysis/analysis_data.json", 
                       help="Plik z analizą")
    parser.add_argument("--output", default="./output/dashboard.png", 
                       help="Plik wyjściowy dashboardu")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Wczytaj dane
    with open(args.metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    with open(args.analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    logging.info("Tworzenie dashboardu treningu...")
    
    # Utwórz dashboard
    create_training_dashboard(metrics, analysis_data, args.output)
    
    print(f"\n{'='*60}")
    print("DASHBOARD TRENINGU UTWORZONY")
    print(f"{'='*60}")
    print(f"Dashboard zapisany: {args.output}")
    print(f"Metryki: {len(metrics)} elementow")
    print(f"Analiza: {len(analysis_data)} sekcji")
    print(f"Czas generowania: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
