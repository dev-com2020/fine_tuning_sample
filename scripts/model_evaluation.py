#!/usr/bin/env python3
"""
Narzędzia do ewaluacji modeli po fine-tuningu
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def setup_logging():
    """Konfiguruje logowanie"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """Wczytuje wyniki ewaluacji"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Oblicza BLEU score (uproszczona wersja)"""
    from collections import Counter
    
    def get_ngrams(text: str, n: int) -> List[str]:
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    if not reference or not candidate:
        return 0.0
    
    # 1-gram
    ref_1gram = Counter(get_ngrams(reference, 1))
    cand_1gram = Counter(get_ngrams(candidate, 1))
    
    # Precision
    matches = sum((ref_1gram & cand_1gram).values())
    total = sum(cand_1gram.values())
    
    if total == 0:
        return 0.0
    
    precision = matches / total
    
    # Brevity penalty
    if len(candidate.split()) <= len(reference.split()):
        bp = np.exp(1 - len(reference.split()) / len(candidate.split()))
    else:
        bp = 1.0
    
    return bp * precision

def calculate_rouge_score(reference: str, candidate: str) -> float:
    """Oblicza ROUGE-L score (uproszczona wersja)"""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    if not reference or not candidate:
        return 0.0
    
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    lcs = lcs_length(ref_words, cand_words)
    
    if len(ref_words) == 0 or len(cand_words) == 0:
        return 0.0
    
    precision = lcs / len(cand_words)
    recall = lcs / len(ref_words)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_responses(results: Dict[str, Any]) -> Dict[str, float]:
    """Ewaluuje odpowiedzi modelu"""
    responses = results['results']
    
    bleu_scores = []
    rouge_scores = []
    response_times = []
    length_ratios = []
    
    for result in responses:
        expected = result['expected'].strip()
        generated = result['generated'].strip() if result['generated'] else ""
        
        # BLEU score
        bleu = calculate_bleu_score(expected, generated)
        bleu_scores.append(bleu)
        
        # ROUGE score
        rouge = calculate_rouge_score(expected, generated)
        rouge_scores.append(rouge)
        
        # Response time
        response_times.append(result['time'])
        
        # Length ratio
        if len(expected) > 0:
            ratio = len(generated) / len(expected)
            length_ratios.append(ratio)
    
    metrics = {
        'avg_bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'avg_rouge': np.mean(rouge_scores) if rouge_scores else 0.0,
        'avg_response_time': np.mean(response_times) if response_times else 0.0,
        'avg_length_ratio': np.mean(length_ratios) if length_ratios else 0.0,
        'total_samples': len(responses),
        'bleu_scores': bleu_scores,
        'rouge_scores': rouge_scores,
        'response_times': response_times,
        'length_ratios': length_ratios
    }
    
    return metrics

def create_evaluation_plots(metrics: Dict[str, Any], output_dir: str) -> None:
    """Tworzy wykresy ewaluacji"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ustawienia stylu
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Wykres 1: Rozkład BLEU scores
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['bleu_scores'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(metrics['avg_bleu'], color='red', linestyle='--', 
                label=f'Średnia: {metrics["avg_bleu"]:.3f}')
    plt.xlabel('BLEU Score')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład BLEU Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'bleu_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 2: Rozkład ROUGE scores
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['rouge_scores'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(metrics['avg_rouge'], color='red', linestyle='--', 
                label=f'Średnia: {metrics["avg_rouge"]:.3f}')
    plt.xlabel('ROUGE Score')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład ROUGE Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'rouge_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 3: Czasy odpowiedzi
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['response_times'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(metrics['avg_response_time'], color='red', linestyle='--', 
                label=f'Średnia: {metrics["avg_response_time"]:.2f}s')
    plt.xlabel('Czas odpowiedzi (s)')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład czasów odpowiedzi')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'response_times.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 4: Stosunek długości
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['length_ratios'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(metrics['avg_length_ratio'], color='red', linestyle='--', 
                label=f'Średnia: {metrics["avg_length_ratio"]:.2f}')
    plt.axvline(1.0, color='green', linestyle=':', 
                label='Idealny stosunek (1.0)')
    plt.xlabel('Stosunek długości (wygenerowane/oczekiwane)')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład stosunku długości odpowiedzi')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'length_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 5: Porównanie metryk
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # BLEU vs ROUGE
    axes[0, 0].scatter(metrics['bleu_scores'], metrics['rouge_scores'], alpha=0.6)
    axes[0, 0].set_xlabel('BLEU Score')
    axes[0, 0].set_ylabel('ROUGE Score')
    axes[0, 0].set_title('BLEU vs ROUGE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Czas vs jakość (BLEU)
    axes[0, 1].scatter(metrics['response_times'], metrics['bleu_scores'], alpha=0.6)
    axes[0, 1].set_xlabel('Czas odpowiedzi (s)')
    axes[0, 1].set_ylabel('BLEU Score')
    axes[0, 1].set_title('Czas vs Jakość (BLEU)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Długość vs jakość
    axes[1, 0].scatter(metrics['length_ratios'], metrics['bleu_scores'], alpha=0.6)
    axes[1, 0].set_xlabel('Stosunek długości')
    axes[1, 0].set_ylabel('BLEU Score')
    axes[1, 0].set_title('Długość vs Jakość')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Podsumowanie metryk
    metric_names = ['BLEU', 'ROUGE', 'Czas (s)', 'Stosunek długości']
    metric_values = [
        metrics['avg_bleu'], 
        metrics['avg_rouge'], 
        metrics['avg_response_time'], 
        metrics['avg_length_ratio']
    ]
    
    bars = axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_title('Średnie wartości metryk')
    axes[1, 1].set_ylabel('Wartość')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Wykresy zapisane w katalogu: {output_path}")

def generate_evaluation_report(results: Dict[str, Any], metrics: Dict[str, Any], 
                             output_path: str) -> None:
    """Generuje raport ewaluacji"""
    
    report = f"""
# Raport ewaluacji modelu

## Podsumowanie
- **Model**: {results['model']}
- **Data ewaluacji**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Liczba próbek**: {results['dataset_size']}
- **Całkowity czas**: {results['total_time']:.2f}s
- **Średni czas na próbkę**: {results['avg_time_per_sample']:.2f}s

## Metryki jakości
- **Średni BLEU Score**: {metrics['avg_bleu']:.3f}
- **Średni ROUGE Score**: {metrics['avg_rouge']:.3f}
- **Średni czas odpowiedzi**: {metrics['avg_response_time']:.2f}s
- **Średni stosunek długości**: {metrics['avg_length_ratio']:.3f}

## Analiza statystyczna
- **Min BLEU**: {min(metrics['bleu_scores']):.3f}
- **Max BLEU**: {max(metrics['bleu_scores']):.3f}
- **Std BLEU**: {np.std(metrics['bleu_scores']):.3f}

- **Min ROUGE**: {min(metrics['rouge_scores']):.3f}
- **Max ROUGE**: {max(metrics['rouge_scores']):.3f}
- **Std ROUGE**: {np.std(metrics['rouge_scores']):.3f}

## Wnioski
"""
    
    # Dodaj wnioski na podstawie metryk
    if metrics['avg_bleu'] > 0.3:
        report += "- Model osiąga dobre wyniki BLEU (>0.3)\n"
    elif metrics['avg_bleu'] > 0.1:
        report += "- Model osiąga umiarkowane wyniki BLEU (0.1-0.3)\n"
    else:
        report += "- Model osiąga niskie wyniki BLEU (<0.1) - wymaga poprawy\n"
    
    if metrics['avg_rouge'] > 0.4:
        report += "- Model osiąga dobre wyniki ROUGE (>0.4)\n"
    elif metrics['avg_rouge'] > 0.2:
        report += "- Model osiąga umiarkowane wyniki ROUGE (0.2-0.4)\n"
    else:
        report += "- Model osiąga niskie wyniki ROUGE (<0.2) - wymaga poprawy\n"
    
    if metrics['avg_response_time'] < 2.0:
        report += "- Model jest szybki (<2s na odpowiedź)\n"
    elif metrics['avg_response_time'] < 5.0:
        report += "- Model ma umiarkowaną szybkość (2-5s na odpowiedź)\n"
    else:
        report += "- Model jest wolny (>5s na odpowiedź) - wymaga optymalizacji\n"
    
    if 0.8 <= metrics['avg_length_ratio'] <= 1.2:
        report += "- Model generuje odpowiedzi o odpowiedniej długości\n"
    elif metrics['avg_length_ratio'] < 0.8:
        report += "- Model generuje zbyt krótkie odpowiedzi\n"
    else:
        report += "- Model generuje zbyt długie odpowiedzi\n"
    
    # Zapisz raport
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info(f"Raport zapisany: {output_path}")

def compare_models(results_paths: List[str], output_path: str) -> None:
    """Porównuje wyniki wielu modeli"""
    logging.info("Porównywanie modeli")
    
    model_results = {}
    
    for results_path in results_paths:
        results = load_evaluation_results(results_path)
        metrics = evaluate_responses(results)
        
        model_name = results['model']
        model_results[model_name] = {
            'results': results,
            'metrics': metrics
        }
    
    # Przygotuj dane do porównania
    comparison_data = []
    
    for model_name, data in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'BLEU': data['metrics']['avg_bleu'],
            'ROUGE': data['metrics']['avg_rouge'],
            'Czas (s)': data['metrics']['avg_response_time'],
            'Stosunek długości': data['metrics']['avg_length_ratio'],
            'Liczba próbek': data['results']['dataset_size']
        })
    
    # Zapisz porównanie jako CSV
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Utwórz wykres porównawczy
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = df['Model']
    
    # BLEU scores
    axes[0, 0].bar(models, df['BLEU'])
    axes[0, 0].set_title('Porównanie BLEU Scores')
    axes[0, 0].set_ylabel('BLEU Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # ROUGE scores
    axes[0, 1].bar(models, df['ROUGE'])
    axes[0, 1].set_title('Porównanie ROUGE Scores')
    axes[0, 1].set_ylabel('ROUGE Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Czasy odpowiedzi
    axes[1, 0].bar(models, df['Czas (s)'])
    axes[1, 0].set_title('Porównanie czasów odpowiedzi')
    axes[1, 0].set_ylabel('Czas (s)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Stosunki długości
    axes[1, 1].bar(models, df['Stosunek długości'])
    axes[1, 1].set_title('Porównanie stosunków długości')
    axes[1, 1].set_ylabel('Stosunek długości')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Zapisz wykres
    plot_path = Path(output_path).with_suffix('.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Porównanie zapisane: {output_path} i {plot_path}")

def main():
    """Główna funkcja skryptu"""
    parser = argparse.ArgumentParser(description="Narzędzia do ewaluacji modeli")
    parser.add_argument("action", choices=["evaluate", "compare"], 
                       help="Akcja do wykonania")
    parser.add_argument("--results", help="Ścieżka do wyników ewaluacji")
    parser.add_argument("--output-dir", default="./output/evaluation", 
                       help="Katalog wyjściowy")
    parser.add_argument("--report", help="Ścieżka do raportu")
    parser.add_argument("--compare-results", nargs='+', 
                       help="Ścieżki do wyników do porównania")
    parser.add_argument("--compare-output", help="Ścieżka do porównania")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.action == "evaluate":
        if not args.results:
            print("Podaj --results")
            return
        
        results = load_evaluation_results(args.results)
        metrics = evaluate_responses(results)
        
        # Utwórz wykresy
        create_evaluation_plots(metrics, args.output_dir)
        
        # Wygeneruj raport
        if args.report:
            generate_evaluation_report(results, metrics, args.report)
        
        # Wyświetl podsumowanie
        print(f"Model: {results['model']}")
        print(f"Liczba próbek: {results['dataset_size']}")
        print(f"Średni BLEU: {metrics['avg_bleu']:.3f}")
        print(f"Średni ROUGE: {metrics['avg_rouge']:.3f}")
        print(f"Średni czas: {metrics['avg_response_time']:.2f}s")
        print(f"Wykresy zapisane w: {args.output_dir}")
    
    elif args.action == "compare":
        if not args.compare_results or not args.compare_output:
            print("Podaj --compare-results i --compare-output")
            return
        
        compare_models(args.compare_results, args.compare_output)

if __name__ == "__main__":
    main()
