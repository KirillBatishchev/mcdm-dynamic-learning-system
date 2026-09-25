"""
Главный модуль эксперимента

Объединяет все компоненты системы:
- Генерацию данных (MCDM: 1000 сотрудников, 50 модулей)
- Расчёт весов (FUCOM + CRITIC + MEREC)
- Загрузку квартилей KTST
- Запуск сценариев S1-S6
- Проверку гипотез H1-H4
- Анализ чувствительности
- Сохранение результатов

H1, H3, H4 - MCDM (на синтетических данных)
H2 - KTST (отдельно, на FORGET-SE)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import time

from data_generation import generate_employees, generate_modules
from weighting import calculate_all_weights
from scipy.stats import mannwhitneyu
from scenarios import (
    run_all_scenarios, run_scenario,
    t_test, shapiro_test
)
from ktst_model import load_quartiles


# КОНСТАНТЫ

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

# Параметры эксперимента
N_EMPLOYEES = 1000
N_MODULES = 50
SEED = 42

# Параметры для быстрых подвыборок
N_SAMPLE_H3_H4 = 50
N_SAMPLE_SENSITIVITY = 100

# Сценарии
SCENARIOS = [
    'S1_Random',
    'S2_GreedyTime',
    'S3_GreedyGain',
    'S4_VIKOR',
    'S5_TOPSIS',
    'S6_WSA',
]

# Основной метод (для сравнения)
MAIN_SCENARIO = 'S4_VIKOR'

# Базовые сценарии (для H1)
BASELINES = ['S1_Random', 'S2_GreedyTime', 'S3_GreedyGain',
             'S5_TOPSIS', 'S6_WSA']


# ПОДГОТОВКА

def setup_directories():
    """Создаёт директории для результатов"""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_quartiles_if_available():
    """Загружает квартили KTST, если они существуют"""
    quartiles_path = MODELS_DIR / 'quartiles.json'
    
    if quartiles_path.exists():
        q25, q50, q75 = load_quartiles(quartiles_path)
        print(f"  Квартили KTST: q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}")
        return (q25, q50, q75)
    else:
        print(f"  Квартили KTST не найдены, fallback")
        return (0.40, 0.60, 0.85)


# ГЛАВНЫЙ ЭКСПЕРИМЕНТ

def run_experiment(n_employees=N_EMPLOYEES, n_modules=N_MODULES, seed=SEED):
    """
    Запускает главный эксперимент (MCDM)
    
    Возвращает:
    (results, hypotheses, sensitivity, employees, modules, weights, quartiles)
    """

    print(f" ! Основной эксперимент (N={n_employees}) ! ")

    
    # 1. Генерация данных
    print("\n--- Генерация данных ---")
    employees = generate_employees(n=n_employees, seed=seed)
    modules = generate_modules(n=n_modules, seed=seed)
    print(f"   Сотрудников: {len(employees)}")
    print(f"   Модулей: {len(modules)}")
    print(f"   Профессиональных: {(modules['type'] == 'professional').sum()}")
    print(f"   Личных: {(modules['type'] == 'personal').sum()}")
    
    # 2. Расчёт весов
    print("\n--- Расчёт весов критериев ---")
    weights_dict = calculate_all_weights(modules, method='hybrid')
    weights = weights_dict['hybrid']
    print(f"   Критериев: {len(weights_dict['criterion_names'])}")
    print(f"   Гибридные веса: {np.round(weights, 4)}")
    
    # 3. Загрузка квартилей KTST
    print("\n--- Загрузка квартилей KTST ---")
    quartiles = load_quartiles_if_available()
    
    # 4. Запуск сценариев
    print("\n--- Запуск сценариев S1-S6 ---")
    results = run_all_scenarios(
        employees, modules, weights,
        quartiles=quartiles,
        weights_for_gas=weights,
        seed=seed
    )
    print(f"   Всего результатов: {len(results)}")
    
    # 5. Проверка H1
    print("\n--- Проверка H1 (качество MCDM) ---")
    h1 = check_h1(results)
    
    # 6. Анализ чувствительности
    print(f"\n--- Анализ чувствительности (N={N_SAMPLE_SENSITIVITY}) ---")
    sensitivity = sensitivity_analysis(
        employees.head(N_SAMPLE_SENSITIVITY), modules, weights,
        quartiles=quartiles, seed=seed
    )
    
    return (results, h1, sensitivity,
            employees, modules, weights, quartiles)


# ПРОВЕРКА H1 (КАЧЕСТВО MCDM)

def cohens_d(group_a, group_b):
    """Cohen's d — размер эффекта."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std

def mann_whitney_test(group_a, group_b):
    """
    U-критерий Манна-Уитни (односторонний).
    
    Проверяет: group_a > group_b.
    """
    u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='greater')
    return float(u_stat), float(p_value)

def check_h1(results):
    """
    Проверяет гипотезу H1
    
    H1: S4 (VIKOR) статистически значимо лучше базовых по CS
    
    Проверяется по двум метрикам: GAS и CS
    """
    hypotheses = []
    
    # Извлечение метрик по сценариям
    gas_by_scenario = {
        s: results[results['scenario'] == s]['GAS'].values
        for s in SCENARIOS
    }
    cs_by_scenario = {
        s: results[results['scenario'] == s]['CS'].values
        for s in SCENARIOS
    }
    
    s4_gas = gas_by_scenario[MAIN_SCENARIO]
    s4_cs = cs_by_scenario[MAIN_SCENARIO]
    
    mean_s4_gas = float(np.mean(s4_gas))
    mean_s4_cs = float(np.mean(s4_cs))
    
    # --- Проверка по GAS ---
    for baseline in BASELINES:
        baseline_gas = gas_by_scenario[baseline]
        mean_baseline = float(np.mean(baseline_gas))
        
        # Проверка нормальности
        _, p_shapiro_s4 = shapiro_test(s4_gas)
        _, p_shapiro_bl = shapiro_test(baseline_gas)
        normal_ok = (p_shapiro_s4 > 0.05) and (p_shapiro_bl > 0.05)
        
        # Выбор теста
        if normal_ok:
            # t-тест (двусторонний)
            t_stat, p_value = t_test(s4_gas, baseline_gas)
            test_name = 't-test'
            statistic = t_stat
            # Одностороннее p-value
            p_one = p_value / 2 if t_stat > 0 else 1.0
        else:
            # U-критерий Манна-Уитни (односторонний)
            u_stat, p_value = mannwhitneyu(s4_gas, baseline_gas, alternative='greater')
            test_name = 'Mann-Whitney'
            statistic = u_stat
            p_one = p_value  # уже одностороннее
        
        s4_better = mean_s4_gas > mean_baseline
        significant = (p_one < 0.05) and s4_better
        
        hypotheses.append({
            'hypothesis': 'H1',
            'metric': 'GAS',
            'comparison': f'{MAIN_SCENARIO} vs {baseline}',
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_one),
            'significant': bool(significant),
            'main_better': bool(s4_better),
            'mean_main': mean_s4_gas,
            'mean_baseline': mean_baseline,
            'delta': mean_s4_gas - mean_baseline,
        })
    
    # --- Проверка по CS ---
    for baseline in BASELINES:
        baseline_cs = cs_by_scenario[baseline]
        mean_baseline = float(np.mean(baseline_cs))
        
        # Проверка нормальности
        _, p_shapiro_s4 = shapiro_test(s4_cs)
        _, p_shapiro_bl = shapiro_test(baseline_cs)
        normal_ok = (p_shapiro_s4 > 0.05) and (p_shapiro_bl > 0.05)
        
        # Выбор теста
        if normal_ok:
            # t-тест (двусторонний)
            t_stat, p_value = t_test(s4_cs, baseline_cs)
            test_name = 't-test'
            statistic = t_stat
            p_one = p_value / 2 if t_stat > 0 else 1.0
        else:
            # U-критерий Манна-Уитни (односторонний)
            u_stat, p_value = mannwhitneyu(s4_cs, baseline_cs, alternative='greater')
            test_name = 'Mann-Whitney'
            statistic = u_stat
            p_one = p_value
        
        s4_better = mean_s4_cs > mean_baseline
        significant = (p_one < 0.05) and s4_better
        d = cohens_d(s4_cs, baseline_cs)
        
        hypotheses.append({
            'hypothesis': 'H1',
            'metric': 'CS',
            'comparison': f'{MAIN_SCENARIO} vs {baseline}',
            'test': test_name,
            'cohens_d': float(d),
            'statistic': float(statistic),
            'p_value': float(p_one),
            'significant': bool(significant),
            'main_better': bool(s4_better),
            'mean_main': mean_s4_cs,
            'mean_baseline': mean_baseline,
            'delta': mean_s4_cs - mean_baseline,
        })
        
    return pd.DataFrame(hypotheses)


# ПРОВЕРКА H3 (УСТОЙЧИВОСТЬ К ν)

def check_h3(employees, modules, weights, quartiles,
             nu_values=None, seed=SEED):
    """
    H3: устойчивость S4 к параметру v.
    
    Возвращает DataFrame с результатами.
    """
    if nu_values is None:
        nu_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    results = []
    
    for nu in nu_values:
        gas_values = []
        cs_values = []
        
        for _, employee in employees.iterrows():
            result = run_scenario(
                employee, modules, weights,
                method='vikor', use_ktst=False,
                quartiles=quartiles, v=nu,
                rng=np.random.default_rng(seed),
                weights_for_gas=weights
            )
            gas_values.append(result['GAS'])
            cs_values.append(result['CS'])
        
        results.append({
            'nu': nu,
            'mean_GAS': float(np.mean(gas_values)),
            'std_GAS': float(np.std(gas_values)),
            'mean_CS': float(np.mean(cs_values)),
            'std_CS': float(np.std(cs_values)),
        })
    
    df = pd.DataFrame(results)
    
    # Разброс
    max_gas = df['mean_GAS'].max()
    min_gas = df['mean_GAS'].min()
    spread_gas = (max_gas - min_gas) / max_gas * 100 if max_gas > 0 else 0.0
    
    max_cs = df['mean_CS'].max()
    min_cs = df['mean_CS'].min()
    spread_cs = (max_cs - min_cs) / max_cs * 100 if max_cs > 0 else 0.0
    
    df['spread_GAS_percent'] = spread_gas
    df['spread_CS_percent'] = spread_cs
    df['h3_confirmed'] = (spread_gas <= 10.0) and (spread_cs <= 10.0)
    
    return df


# ПРОВЕРКА H4 (ВЫЧИСЛИТЕЛЬНАЯ СЛОЖНОСТЬ)

def check_h4(employees, modules, weights, quartiles,
             n_values=None, n_sample=None, seed=SEED):
    """
    H4: время работы растёт линейно с числом модулей
    
    Возвращает: DataFrame с результатами
    """
    if n_values is None:
        n_values = [10, 25, 50, 100, 500]
    if n_sample is None:
        n_sample = min(N_SAMPLE_H3_H4, len(employees))
    
    results = []
    sample_employees = employees.head(n_sample)
    
    for n in n_values:
        if n <= len(modules):
            modules_subset = modules.head(n).copy()
        else:
            repeat_factor = (n // len(modules)) + 1
            modules_subset = pd.concat([modules] * repeat_factor,
                                       ignore_index=True).head(n)
            modules_subset['module_id'] = range(len(modules_subset))
        
        start = time.time()
        for _, employee in sample_employees.iterrows():
            run_scenario(
                employee, modules_subset, weights,
                method='vikor', use_ktst=False,
                quartiles=quartiles,
                rng=np.random.default_rng(seed),
                weights_for_gas=weights
            )
        elapsed = time.time() - start
        
        results.append({
            'n_modules': n,
            'time_seconds': float(elapsed),
            'time_per_employee': float(elapsed / n_sample),
        })
    
    df = pd.DataFrame(results)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['n_modules'], df['time_seconds']
    )
    
    df['r_squared'] = float(r_value ** 2)
    df['slope'] = float(slope)
    df['h4_confirmed'] = (r_value ** 2) >= 0.95
    
    return df


# АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ

def sensitivity_analysis(employees, modules, weights,
                          quartiles, seed=SEED):
    """Анализ чувствительности к порогам, шуму, размеру группы"""
    results = []
    q25, q50, q75 = quartiles
    
    # --- 1. Пороги KTST ---
    print(" Пороги KTST...")
    for delta in [-0.05, 0.0, 0.05]:
        q25_mod = np.clip(q25 + delta, 0.01, 0.99)
        q50_mod = np.clip(q50 + delta, 0.01, 0.99)
        q75_mod = np.clip(q75 + delta, 0.01, 0.99)
        
        gas_values = []
        for _, employee in employees.iterrows():
            result = run_scenario(
                employee, modules, weights,
                method='vikor', use_ktst=True,
                quartiles=(q25_mod, q50_mod, q75_mod),
                rng=np.random.default_rng(seed),
                weights_for_gas=weights
            )
            gas_values.append(result['GAS'])
        
        results.append({
            'type': 'threshold',
            'param_name': 'delta',
            'param_value': delta,
            'mean_GAS': float(np.mean(gas_values)),
            'std_GAS': float(np.std(gas_values)),
        })
    
    # --- 2. Шум в данных ---
    print(" Шум в данных...")
    for sigma in [0.05, 0.10, 0.20]:
        rng = np.random.default_rng(seed)
        employees_noisy = employees.copy()
        
        for col in ['knowledge', 'digital_literacy']:
            noise = rng.normal(0, sigma, len(employees))
            employees_noisy[col] = np.clip(employees_noisy[col] + noise, 0, 1)
        
        gas_values = []
        for _, employee in employees_noisy.iterrows():
            result = run_scenario(
                employee, modules, weights,
                method='vikor', use_ktst=False,
                quartiles=quartiles,
                rng=np.random.default_rng(seed),
                weights_for_gas=weights
            )
            gas_values.append(result['GAS'])
        
        results.append({
            'type': 'noise',
            'param_name': 'sigma',
            'param_value': sigma,
            'mean_GAS': float(np.mean(gas_values)),
            'std_GAS': float(np.std(gas_values)),
        })
    
    # --- 3. Размер группы ---
    print(" Размер группы...")
    for n in [25, 50, 100, 200]:
        if n <= len(employees):
            subset = employees.head(n)
        else:
            repeat_factor = (n // len(employees)) + 1
            subset = pd.concat([employees] * repeat_factor,
                               ignore_index=True).head(n)
            subset['employee_id'] = range(len(subset))
        
        gas_values = []
        for _, employee in subset.iterrows():
            result = run_scenario(
                employee, modules, weights,
                method='vikor', use_ktst=False,
                quartiles=quartiles,
                rng=np.random.default_rng(seed),
                weights_for_gas=weights
            )
            gas_values.append(result['GAS'])
        
        results.append({
            'type': 'group_size',
            'param_name': 'n',
            'param_value': n,
            'mean_GAS': float(np.mean(gas_values)),
            'std_GAS': float(np.std(gas_values)),
        })
    
    return pd.DataFrame(results)


# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

def save_results(results, h1, h3, h4, sensitivity):
    """Сохраняет все результаты в CSV"""
    setup_directories()
    
    # Основные результаты
    results_save = results.drop(columns=['profile_history', 'time_history'],
                                 errors='ignore')
    results_save.to_csv(TABLES_DIR / 'results_full.csv', index=False)
    
    # Сводка по сценариям
    summary = results.groupby('scenario').agg(
        mean_GAS=('GAS', 'mean'),
        std_GAS=('GAS', 'std'),
        var_GAS=('GAS', 'var'),
        mean_CS=('CS', 'mean'),
        std_CS=('CS', 'std'),
        mean_ME=('ME', 'mean'),
        mean_TE=('TE', 'mean'),
        mean_time=('time_used', 'mean'),
        std_time=('time_used', 'std'),
        mean_modules=('n_modules', 'mean'),
    ).round(4).reset_index()
    summary.to_csv(TABLES_DIR / 'summary_scenarios.csv', index=False)
    
    # Гипотезы
    h1.to_csv(TABLES_DIR / 'hypotheses_H1.csv', index=False)
    h3.to_csv(TABLES_DIR / 'hypotheses_H3.csv', index=False)
    h4.to_csv(TABLES_DIR / 'hypotheses_H4.csv', index=False)
    
    # Чувствительность
    sensitivity.to_csv(TABLES_DIR / 'sensitivity.csv', index=False)
    
    print(f"\nРезультаты сохранены в {TABLES_DIR}")


def print_summary(results, h1, h3, h4, sensitivity):
    """Выводит сводку эксперимента."""
    print("--- СВОДКА ЭКСПЕРИМЕНТА ---")
    
    # Основные результаты
    print("\n--- Результаты по сценариям ---")
    summary = results.groupby('scenario').agg(
        mean_GAS=('GAS', 'mean'),
        mean_CS=('CS', 'mean'),
        mean_ME=('ME', 'mean'),
        mean_TE=('TE', 'mean'),
        mean_time=('time_used', 'mean'),
        mean_modules=('n_modules', 'mean'),
    ).round(4)
    print(summary)
    
    # H1 по GAS
    print("\n--- H1: S4 vs базовые (GAS) ---")
    print(f"  {'Сравнение':<30} | {'S4':>8} | {'baseline':>8} | "
        f"{'Δ':>8} | {'p-value':>10} | {'Значимо':>8}")
    print("-" * 90)

    h1_gas = h1[h1['metric'] == 'GAS']
    for _, row in h1_gas.iterrows():
        sig = "+" if row['significant'] else "-"
        print(f"  {row['comparison']:<30} | "
            f"{row['mean_main']:>8.4f} | "
            f"{row['mean_baseline']:>8.4f} | "
            f"{row['delta']:>+8.4f} | "
            f"{row['p_value']:>10.6f} | "
            f"{sig:>8}")

    # H1 по CS
    print("\n--- H1: S4 vs базовые (CS) ---")
    print(f"  {'Сравнение':<30} | {'S4':>8} | {'baseline':>8} | "
        f"{'Δ':>8} | {'p-value':>10} | {'Cohen d':>8} | {'Значимо':>8}")
    print("-" * 105)

    h1_cs = h1[h1['metric'] == 'CS']
    for _, row in h1_cs.iterrows():
        sig = "+" if row['significant'] else "-"
        d = row.get('cohens_d', 0.0)
        print(f"  {row['comparison']:<30} | "
            f"{row['mean_main']:>8.4f} | "
            f"{row['mean_baseline']:>8.4f} | "
            f"{row['delta']:>+8.4f} | "
            f"{row['p_value']:>10.6f} | "
            f"{d:>8.3f} | "
            f"{sig:>8}")

    # H3
    print("\n--- H3: устойчивость к v ---")
    print(f"  {'v':>6} | {'Средний GAS':>12} | {'Std GAS':>10} | "
        f"{'Средний CS':>12} | {'Std CS':>10}")
    print("-" * 65)

    for _, row in h3.iterrows():
        print(f"  {row['nu']:>6.2f} | "
            f"{row['mean_GAS']:>12.4f} | "
            f"{row['std_GAS']:>10.4f} | "
            f"{row['mean_CS']:>12.4f} | "
            f"{row['std_CS']:>10.4f}")

    print("-" * 65)

    # Мин/макс
    idx_min_gas = h3['mean_GAS'].idxmin()
    idx_max_gas = h3['mean_GAS'].idxmax()
    idx_min_cs = h3['mean_CS'].idxmin()
    idx_max_cs = h3['mean_CS'].idxmax()

    print(f"\n  Минимальный GAS: {h3.loc[idx_min_gas, 'mean_GAS']:.4f} "
        f"(v = {h3.loc[idx_min_gas, 'nu']})")
    print(f"  Максимальный GAS: {h3.loc[idx_max_gas, 'mean_GAS']:.4f} "
        f"(v = {h3.loc[idx_max_gas, 'nu']})")
    print(f"  Минимальный CS: {h3.loc[idx_min_cs, 'mean_CS']:.4f} "
        f"(v = {h3.loc[idx_min_cs, 'nu']})")
    print(f"  Максимальный CS: {h3.loc[idx_max_cs, 'mean_CS']:.4f} "
        f"(v = {h3.loc[idx_max_cs, 'nu']})")

    # Разброс
    spread_gas = h3['spread_GAS_percent'].iloc[0]
    spread_cs = h3['spread_CS_percent'].iloc[0]

    print(f"\n  Разброс GAS: {spread_gas:.4f}%")
    print(f"  Разброс CS: {spread_cs:.4f}%")
    print(f"  Порог: 10%")

    # Итог
    sig = "подтверждена" if (spread_gas <= 10) and (spread_cs <= 10) else "НЕ подтверждена"
    print(f"  H3: {sig}")
    
    # H4
    print("\n--- H4: вычислительная сложность ---")
    r_sq = h4['r_squared'].iloc[0]
    print(f"  R² линейной регрессии: {r_sq:.4f}")
    sig = "подтверждена" if r_sq >= 0.95 else "НЕ подтверждена"
    print(f"  H4: {sig}")
    
    print("\n--- Анализ чувствительности ---")
    print(f"  {'Тип':<12} | {'Параметр':<10} | {'Значение':>10} | "
          f"{'Средний GAS':>12} | {'Std GAS':>10}")
    print("-" * 70)
    
    for _, row in sensitivity.iterrows():
        print(f"  {row['type']:<12} | "
              f"{row['param_name']:<10} | "
              f"{row['param_value']:>10.2f} | "
              f"{row['mean_GAS']:>12.4f} | "
              f"{row['std_GAS']:>10.4f}")
    
    # Разброс по типам
    print()
    for t in sensitivity['type'].unique():
        subset = sensitivity[sensitivity['type'] == t]
        gas_min = subset['mean_GAS'].min()
        gas_max = subset['mean_GAS'].max()
        spread = (gas_max - gas_min) / gas_max * 100 if gas_max > 0 else 0.0
        print(f"  {t:<12}: разброс GAS = {spread:.2f}%")


# ОСНОВНОЙ БЛОК

if __name__ == '__main__':
    start_time = time.time()
    
    setup_directories()
    
    print(f" ! Запуск эксперимента (N={N_EMPLOYEES}, N_MODULES={N_MODULES}) ! ")
    
    # 1. Главный эксперимент (MCDM + H1)
    (results, h1, sensitivity,
     employees, modules, weights, quartiles) = run_experiment(
        n_employees=N_EMPLOYEES,
        n_modules=N_MODULES,
        seed=SEED
    )
    
    # 2. H3 на подвыборке
    print(f"\n--- Проверка H3 (устойчивость к v) на N={N_SAMPLE_H3_H4} ---")
    h3 = check_h3(
        employees.head(N_SAMPLE_H3_H4), modules, weights, quartiles, seed=SEED
    )
    
    # 3. H4 на подвыборке
    print(f"\n--- Проверка H4 (вычислительная сложность) на N={N_SAMPLE_H3_H4} ---")
    h4 = check_h4(
        employees.head(N_SAMPLE_H3_H4), modules, weights, quartiles, seed=SEED
    )
    
    # 4. Сохранение
    print("\n--- Сохранение результатов ---")
    save_results(results, h1, h3, h4, sensitivity)
    
    # 5. Сводка
    print_summary(results, h1, h3, h4, sensitivity)
    
    total_time = time.time() - start_time
    print(f"\nЭксперимент завершён за {total_time:.1f} сек "
          f"({total_time / 60:.1f} мин)")
    print(f"   Таблицы: {TABLES_DIR}")
    print(f"   Графики: {FIGURES_DIR}")