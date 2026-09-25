"""
Модуль визуализации результатов экспериментов

Реализует:
- Boxplot GAS и CS по сценариям S1–S6
- Scatter GAS vs Time
- График зависимости GAS от v (H3)
- График зависимости времени от N (H4)
- График анализа чувствительности
- График динамики обучения KTST

Запускается после experiment.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def ensure_dirs():
    """Создаёт директории для графиков"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Загружает результаты экспериментов из CSV"""
    results_path = TABLES_DIR / 'results_full.csv'
    h1_path = TABLES_DIR / 'hypotheses_H1.csv'
    h3_path = TABLES_DIR / 'hypotheses_H3.csv'
    h4_path = TABLES_DIR / 'hypotheses_H4.csv'
    sensitivity_path = TABLES_DIR / 'sensitivity.csv'
    
    results = pd.read_csv(results_path) if results_path.exists() else None
    h1 = pd.read_csv(h1_path) if h1_path.exists() else None
    h3 = pd.read_csv(h3_path) if h3_path.exists() else None
    h4 = pd.read_csv(h4_path) if h4_path.exists() else None
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else None
    
    return results, h1, h3, h4, sensitivity


# 1. BOXPLOT GAS И CS

def plot_boxplot_gas_cs(results):
    """Boxplot GAS и CS по сценариям S1–S6"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = results['scenario'].unique()
    
    # GAS
    gas_data = [results[results['scenario'] == s]['GAS'].values 
                for s in scenarios]
    bp1 = axes[0].boxplot(gas_data, labels=scenarios, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'),
                          medianprops=dict(color='red', linewidth=2))
    axes[0].set_title('Распределение GAS по сценариям', fontsize=12)
    axes[0].set_ylabel('GAS')
    axes[0].set_xlabel('Сценарий')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # CS
    cs_data = [results[results['scenario'] == s]['CS'].values 
               for s in scenarios]
    bp2 = axes[1].boxplot(cs_data, labels=scenarios, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen'),
                          medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('Распределение CS по сценариям', fontsize=12)
    axes[1].set_ylabel('CS')
    axes[1].set_xlabel('Сценарий')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    path = FIGURES_DIR / 'boxplot_gas_cs.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {path}")


# 2. ГРАФИК H3 (УСТОЙЧИВОСТЬ К ν)

def plot_h3(h3):
    """График зависимости GAS и CS от v"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GAS
    axes[0].plot(h3['nu'], h3['mean_GAS'], 'o-', color='blue', linewidth=2,
                 markersize=8)
    axes[0].fill_between(h3['nu'], 
                         h3['mean_GAS'] - h3['std_GAS'],
                         h3['mean_GAS'] + h3['std_GAS'],
                         alpha=0.3, color='blue')
    axes[0].set_title('Зависимость GAS от v', fontsize=12)
    axes[0].set_xlabel('Параметр стратегии v')
    axes[0].set_ylabel('Средний GAS')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # CS
    axes[1].plot(h3['nu'], h3['mean_CS'], 's-', color='green', linewidth=2,
                 markersize=8)
    axes[1].fill_between(h3['nu'],
                         h3['mean_CS'] - h3['std_CS'],
                         h3['mean_CS'] + h3['std_CS'],
                         alpha=0.3, color='green')
    axes[1].set_title('Зависимость CS от v', fontsize=12)
    axes[1].set_xlabel('Параметр стратегии v')
    axes[1].set_ylabel('Средний CS')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    plt.tight_layout()
    path = FIGURES_DIR / 'h3_nu.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {path}")


# 3. ГРАФИК H4 (ВЫЧИСЛИТЕЛЬНАЯ СЛОЖНОСТЬ)

def plot_h4(h4):
    """График зависимости времени от N"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(h4['n_modules'], h4['time_seconds'], 'o-', 
            color='purple', linewidth=2, markersize=10)
    
    # Линейная регрессия
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        h4['n_modules'], h4['time_seconds']
    )
    x_line = np.linspace(h4['n_modules'].min(), h4['n_modules'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color='red', linewidth=1.5,
            label=f'Линейная регрессия (R² = {r_value**2:.4f})')
    
    ax.set_title('Зависимость времени работы от числа модулей', fontsize=12)
    ax.set_xlabel('Число модулей N')
    ax.set_ylabel('Время работы (секунды)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / 'h4_time.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {path}")


# 4. ГРАФИК ЧУВСТВИТЕЛЬНОСТИ

def plot_sensitivity(sensitivity):
    """График анализа чувствительности"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    types = sensitivity['type'].unique()
    
    for i, t in enumerate(types):
        subset = sensitivity[sensitivity['type'] == t]
        axes[i].plot(subset['param_value'], subset['mean_GAS'],
                     'o-', color='darkblue', linewidth=2, markersize=8)
        axes[i].fill_between(subset['param_value'],
                             subset['mean_GAS'] - subset['std_GAS'],
                             subset['mean_GAS'] + subset['std_GAS'],
                             alpha=0.3, color='blue')
        axes[i].set_title(f'Чувствительность: {t}', fontsize=12)
        axes[i].set_xlabel(subset['param_name'].iloc[0])
        axes[i].set_ylabel('Средний GAS')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / 'sensitivity.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {path}")


# 5. ГРАФИК ДИНАМИКИ ОБУЧЕНИЯ KTST

def plot_ktst_training():
    """
    График динамики обучения KTST
    
    Читает логи из models/ktst_training_log.json
    """
    log_path = PROJECT_ROOT / 'models' / 'ktst_training_log.json'
    
    if not log_path.exists():
        print(f"Лог обучения KTST не найден: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = log['epochs']
    losses = log['losses']
    val_aucs = log['val_aucs']
    
    axes[0].plot(epochs, losses, 'o-', color='red', linewidth=2)
    axes[0].set_title('Динамика Loss при обучении KTST', fontsize=12)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, val_aucs, 's-', color='green', linewidth=2)
    axes[1].set_title('Динамика Val AUC при обучении KTST', fontsize=12)
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Val AUC')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / 'ktst_training.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {path}")


# 6. СРАВНЕНИЕ СРЕДНИХ GAS И CS

def plot_mean_comparison(results):
    """Сравнение средних GAS и CS по сценариям (barplot)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    summary = results.groupby('scenario').agg(
        mean_GAS=('GAS', 'mean'),
        std_GAS=('GAS', 'std'),
        mean_CS=('CS', 'mean'),
        std_CS=('CS', 'std'),
    ).reset_index()
    
    # GAS
    colors_gas = ['#d62728' if s == 'S4_VIKOR' else '#1f77b4' 
                  for s in summary['scenario']]
    axes[0].bar(summary['scenario'], summary['mean_GAS'],
                yerr=summary['std_GAS'], capsize=5, color=colors_gas)
    axes[0].set_title('Средний GAS по сценариям', fontsize=12)
    axes[0].set_ylabel('GAS')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # CS
    colors_cs = ['#d62728' if s == 'S4_VIKOR' else '#2ca02c' 
                 for s in summary['scenario']]
    axes[1].bar(summary['scenario'], summary['mean_CS'],
                yerr=summary['std_CS'], capsize=5, color=colors_cs)
    axes[1].set_title('Средний CS по сценариям', fontsize=12)
    axes[1].set_ylabel('CS')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = FIGURES_DIR / 'mean_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {path}")


# ОСНОВНОЙ БЛОК

def main():
    """Запускает все визуализации"""
    print(" ! Визуализация результатов экспериментов ! ")
    
    ensure_dirs()
    
    # Загрузка результатов
    results, h1, h3, h4, sensitivity = load_results()
    
    if results is None:
        print("Результаты не найдены. Запустите experiment.py")
        return
    
    # 1. Boxplot GAS и CS
    print("\n--- Boxplot GAS и CS ---")
    plot_boxplot_gas_cs(results)
    
    # 2. Сравнение средних
    print("\n--- Сравнение средних GAS и CS ---")
    plot_mean_comparison(results)
    
    # 3. H3
    if h3 is not None:
        print("\n--- График H3 (устойчивость к v) ---")
        plot_h3(h3)
    
    # 4. H4
    if h4 is not None:
        print("\n--- График H4 (вычислительная сложность) ---")
        plot_h4(h4)
    
    # 5. Чувствительность
    if sensitivity is not None:
        print("\n--- График чувствительности ---")
        plot_sensitivity(sensitivity)
    
    # 6. KTST
    print("\n--- График обучения KTST ---")
    plot_ktst_training()
    
    print(f"Все графики сохранены в {FIGURES_DIR}")


if __name__ == '__main__':
    main()