"""
Модуль сценариев формирования образовательных траекторий

Реализует:
- Жадный алгоритм с ранжированием на каждом шаге (MCDM-ядро)
- Адаптацию количества заданий на основе KTST
- Обновление профиля сотрудника (включая компоненты выгорания и IntPV)
- Сценарии S1–S6 (S4 - VIKOR, S5 - TOPSIS, S6 - WSA)
- Метрики: GAS, CS, ME, TE, дисперсия, RT, t-критерий, критерий Фишера
"""

import numpy as np
import sys
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
from data_generation import generate_employees, generate_modules
from weighting import calculate_all_weights
from ktst_model import load_quartiles

from ranking import rank_modules, CRITERION_NAMES, CRITERIA_TYPES
from weighting import load_weights


warnings.filterwarnings('ignore')


# КОНСТАНТЫ

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# Соответствие критериев модуля и полей профиля
CRITERION_TO_PROFILE = {
    'delta_prof': 'knowledge',
    'delta_digital': 'digital_literacy',
    'delta_stress': 'stress_resistance',
    'delta_reflex': 'reflexivity',
    'delta_proact': 'proactivity',
    'delta_SI': 'SI',
    'delta_DP': 'DP',
    'delta_RLD': 'RLD',
}

# Целевые значения
TARGET_VALUES = {
    'knowledge': 0.90,
    'digital_literacy': 0.85,
    'stress_resistance': 7.5,
    'reflexivity': 140,
    'proactivity': 30,
    'SI': 10.0,
    'DP': 5.0,
    'RLD': 40.0,
    'IntPV': 3.0,
}

# Масштабные коэффициенты
SCALE_FACTORS = {
    'knowledge': 1.0,
    'digital_literacy': 1.0,
    'stress_resistance': 10.0,
    'reflexivity': 180.0,
    'proactivity': 32.0,
    'SI': 54.0,
    'DP': 30.0,
    'RLD': 48.0,
    'IntPV': 10.0,
}

# Параметры комбинированной метрики
CS_W1 = 0.5    # вес GAS
CS_W2 = 0.25   # вес времени
CS_W3 = 0.25   # вес модулей

CS_MODULES_MAX = 10.0   # максимальное число модулей


# ФОРМУЛА АДАПТАЦИИ (tasks_factor, gamma_factor)

def compute_adaptation(p_success, q25, q50, q75):
    """
    Коэффициенты адаптации KTST
    
    Возвращает: (tasks_factor, gamma_factor)
    """
    if p_success > q75:
        return 0.7, 1.0
    elif p_success > q50:
        return 1.0, 1.0
    elif p_success > q25:
        return 1.3, 1.0
    else:
        return 1.6, 0.9


# ОБНОВЛЕНИЕ ПРОФИЛЯ

def update_profile(profile, module, p_success, module_type, gamma_factor=1.0):
    """
    Обновляет профиль сотрудника после прохождения модуля
    
    Формула (29): P_new = P + Δ_module x γ
    После обновления компонент выгорания пересчитывается IntPV
    """
    profile_new = profile.copy()
    
    gamma = min(1.0, p_success + 0.2) * gamma_factor
    
    # 1. Обновление критериев
    for criterion, profile_key in CRITERION_TO_PROFILE.items():
        gain = module[criterion]
        if gain == 0:
            continue
        
        if criterion in ['delta_prof', 'delta_digital']:
            gamma_used = gamma
        else:
            gamma_used = 0.5
        
        scale = SCALE_FACTORS[profile_key]
        profile_new[profile_key] += gain * scale * gamma_used
    
    # 2. Обрезка базовых критериев
    for key in ['knowledge', 'digital_literacy', 'stress_resistance',
                'reflexivity', 'proactivity']:
        profile_new[key] = min(profile_new[key], TARGET_VALUES[key])
    
    # 3. Обрезка компонент выгорания
    profile_new['SI'] = np.clip(profile_new['SI'], 0, 54)
    profile_new['DP'] = np.clip(profile_new['DP'], 0, 30)
    profile_new['RLD'] = np.clip(profile_new['RLD'], 0, 48)
    
    # 4. Пересчёт IntPV
    profile_new['IntPV'] = (
        4.386
        + 0.1155 * profile_new['SI']
        + 0.1747 * profile_new['DP']
        - 0.0998 * profile_new['RLD']
    )
    
    return profile_new


# МЕТРИКИ

def compute_gas(profile_final, profile_target, weights=None):
    """
    Goal Achievement Score
    
    GAS = 1 - ||weights x (P_final - T)|| / ||weights x T||
    """
    keys = ['knowledge', 'digital_literacy', 'stress_resistance',
            'reflexivity', 'proactivity', 'IntPV']
    
    p_final = np.array([profile_final[k] / SCALE_FACTORS.get(k, 1.0)
                        for k in keys])
    p_target = np.array([profile_target[k] / SCALE_FACTORS.get(k, 1.0)
                         for k in keys])
    
    if weights is None:
        w = np.ones(len(keys))
    else:
        w = np.array(weights)
        if len(w) != len(keys):
            w = np.ones(len(keys))
    
    diff_norm = np.linalg.norm(w * (p_final - p_target))
    target_norm = np.linalg.norm(w * p_target)
    
    gas = 1 - diff_norm / (target_norm + 1e-10)
    return float(np.clip(gas, 0.0, 1.0))


def compute_me(gas, n_modules):
    """
    Module Efficiency (ME) - GAS на один модуль
    
    Показывает, насколько эффективно метод использует учебные модули
    Чем выше - тем лучше
    """
    return float(gas / (n_modules + 1e-10))


def compute_te(gas, time_used):
    """
    Time Efficiency (TE) - GAS на час обучения
    
    Показывает, насколько эффективно метод использует время
    Чем выше - тем лучше
    """
    return float(gas / (time_used + 1e-10))


def compute_cs(gas, time_used, n_modules, time_max=250.0,
               w1=CS_W1, w2=CS_W2, w3=CS_W3):
    """
    Composite Score (CS) - комбинированная метрика
    
    Объединяет:
    - GAS (качество) - вес w1;
    - экономию времени (1 - Time/Time_max) - вес w2;
    - экономию модулей (1 - Modules/Modules_max) - вес w3
    
    Возвращает: CS ∈ [0, 1]
    """
    time_score = 1.0 - min(time_used / time_max, 1.0)
    modules_score = 1.0 - min(n_modules / CS_MODULES_MAX, 1.0)
    
    cs = w1 * gas + w2 * time_score + w3 * modules_score
    return float(np.clip(cs, 0.0, 1.0))


def compute_dispersion(gas_values):
    """Дисперсия GAS внутри группы"""
    return float(np.var(gas_values))


def compute_rt(time_alg, time_linear):
    """Relative Time"""
    return float(time_alg / (time_linear + 1e-10))


def t_test(group_a, group_b):
    """t-критерий Стьюдента"""
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    return float(t_stat), float(p_value)


def shapiro_test(values):
    """Критерий Шапиро-Уилка"""
    if len(values) < 3:
        return 1.0, 1.0
    w_stat, p_value = stats.shapiro(values)
    return float(w_stat), float(p_value)


def fisher_test(var_a, var_b, n_a, n_b):
    """Критерий Фишера для сравнения дисперсий"""
    if var_a > var_b:
        f_stat = var_a / (var_b + 1e-10)
        df1, df2 = n_a - 1, n_b - 1
    else:
        f_stat = var_b / (var_a + 1e-10)
        df1, df2 = n_b - 1, n_a - 1
    
    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2),
                      1 - stats.f.cdf(f_stat, df1, df2))
    return float(f_stat), float(p_value)


# ИНИЦИАЛИЗАЦИЯ

def init_profile(employee):
    """Создаёт словарь профиля из строки DataFrame"""
    return {
        'knowledge': float(employee['knowledge']),
        'digital_literacy': float(employee['digital_literacy']),
        'stress_resistance': float(employee['stress_resistance']),
        'reflexivity': float(employee['reflexivity']),
        'proactivity': float(employee['proactivity']),
        'SI': float(employee['SI']),
        'DP': float(employee['DP']),
        'RLD': float(employee['RLD']),
        'IntPV': float(employee['IntPV']),
    }


def init_target(employee):
    """Создаёт словарь целевого профиля"""
    return {
        'knowledge': float(employee['target_knowledge']),
        'digital_literacy': float(employee['target_digital']),
        'stress_resistance': float(employee['target_stress']),
        'reflexivity': float(employee['target_reflexivity']),
        'proactivity': float(employee['target_proactivity']),
        'SI': TARGET_VALUES['SI'],
        'DP': TARGET_VALUES['DP'],
        'RLD': TARGET_VALUES['RLD'],
        'IntPV': float(employee['target_IntPV']),
    }


def compute_deficits(profile, target):
    """Вычисляет дефициты в нормализованной шкале"""
    deficits = np.zeros(len(CRITERION_NAMES))
    for i, criterion in enumerate(CRITERION_NAMES):
        profile_key = CRITERION_TO_PROFILE[criterion]
        scale = SCALE_FACTORS[profile_key]
        target_val = target[profile_key] / scale
        current_val = profile[profile_key] / scale
        deficits[i] = max(0, target_val - current_val)
    
    return deficits


# KTST-ПРОГНОЗ

def predict_ktst(q25, q50, q75, rng=None):
    """Сэмплирует P_success из распределения квартилей KTST."""
    if rng is None:
        rng = np.random.default_rng()
    
    return float(rng.triangular(q25, q50, q75))


# УНИВЕРСАЛЬНЫЙ СЦЕНАРИЙ

def run_scenario(employee, modules, weights, method='vikor',
                 use_ktst=False, quartiles=None,
                 v=0.5, rng=None, weights_for_gas=None):
    """
    Запуск одного сценария для одного сотрудника
    
    Возвращает: dict с результатами (GAS, CS, ME, TE, время, модули)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Инициализация
    profile = init_profile(employee)
    target = init_target(employee)
    remaining_time = float(employee['available_time'])
    initial_time = remaining_time
    
    selected = []
    available = list(modules.index)
    profile_history = [profile.copy()]
    time_history = [0.0]
    
    # Квартили
    if quartiles is None:
        quartiles = (0.40, 0.60, 0.85)
    q25, q50, q75 = quartiles
    
    # KTST
    p_success_current = 0.7
    gamma_factor = 1.0
    tasks_factor = 1.0
    
    # Цикл
    while remaining_time > 0 and len(available) > 0:
        # 1. Фильтрация по времени
        valid = [i for i in available
                 if modules.iloc[i]['base_time'] * tasks_factor <= remaining_time]
        if not valid:
            break
        
        # 2. Дефициты
        deficits = compute_deficits(profile, target)
        
        # 3. Ранжирование
        best_idx = rank_modules(
            valid, modules, weights, method=method,
            criteria_types=CRITERIA_TYPES,
            criterion_names=CRITERION_NAMES,
            deficits=deficits, rng=rng, v=v
        )
        
        # 4. Применение модуля
        module = modules.iloc[best_idx]
        base_time = float(module['base_time'])
        
        # KTST-адаптация
        if use_ktst:
            p_success_current = predict_ktst(q25, q50, q75, rng)
            tasks_factor, gamma_factor = compute_adaptation(
                p_success_current, q25, q50, q75
            )
            actual_time = base_time * tasks_factor
        else:
            actual_time = base_time
            p_success_current = 0.7
            gamma_factor = 1.0
        
        # Обрезка по времени
        if actual_time > remaining_time:
            actual_time = remaining_time
        
        # 5. Обновление профиля
        profile = update_profile(
            profile, module, p_success_current, module['type'], gamma_factor
        )
        
        # 6. Обновление
        remaining_time -= actual_time
        selected.append(best_idx)
        available.remove(best_idx)
        
        profile_history.append(profile.copy())
        time_history.append(initial_time - remaining_time)
    
    # Метрики
    gas = compute_gas(profile, target, weights=weights_for_gas)
    time_used = initial_time - remaining_time
    n_modules = len(selected)
    
    me = compute_me(gas, n_modules)
    te = compute_te(gas, time_used)
    cs = compute_cs(gas, time_used, n_modules,
                    time_max=initial_time, w1=CS_W1, w2=CS_W2, w3=CS_W3)
    
    return {
        'employee_id': int(employee['employee_id']),
        'selected_modules': selected,
        'final_profile': profile,
        'target_profile': target,
        'time_used': time_used,
        'GAS': gas,
        'ME': me,
        'TE': te,
        'CS': cs,
        'n_modules': n_modules,
        'profile_history': profile_history,
        'time_history': time_history,
    }


# ОБЁРТКИ ДЛЯ СЦЕНАРИЕВ S1–S6

def run_s1_random(employee, modules, weights, **kwargs):
    return run_scenario(employee, modules, weights, method='random',
                        use_ktst=False, **kwargs)


def run_s2_greedy_time(employee, modules, weights, **kwargs):
    return run_scenario(employee, modules, weights, method='greedy_time',
                        use_ktst=False, **kwargs)


def run_s3_greedy_gain(employee, modules, weights, **kwargs):
    return run_scenario(employee, modules, weights, method='greedy_gain',
                        use_ktst=False, **kwargs)


def run_s4_vikor(employee, modules, weights, **kwargs):
    """S4 - VIKOR (основной метод)"""
    return run_scenario(employee, modules, weights, method='vikor',
                        use_ktst=False, **kwargs)


def run_s5_topsis(employee, modules, weights, **kwargs):
    """S5 - TOPSIS (альтернативный MCDM)"""
    return run_scenario(employee, modules, weights, method='topsis',
                        use_ktst=False, **kwargs)


def run_s6_wsa(employee, modules, weights, **kwargs):
    """S6 - WSA (взвешенная сумма)"""
    return run_scenario(employee, modules, weights, method='wsa',
                        use_ktst=False, **kwargs)


SCENARIO_RUNNERS = {
    'S1_Random': run_s1_random,
    'S2_GreedyTime': run_s2_greedy_time,
    'S3_GreedyGain': run_s3_greedy_gain,
    'S4_VIKOR': run_s4_vikor,
    'S5_TOPSIS': run_s5_topsis,
    'S6_WSA': run_s6_wsa,
}


# ЗАПУСК ВСЕХ СЦЕНАРИЕВ

def run_all_scenarios(employees, modules, weights, quartiles=None,
                      weights_for_gas=None, seed=42):
    """Запускает все 6 сценариев для всех сотрудников"""
    rng = np.random.default_rng(seed)
    results = []
    
    for scenario_name, runner in SCENARIO_RUNNERS.items():
        print(f"Запуск {scenario_name}...")
        for _, employee in employees.iterrows():
            result = runner(
                employee, modules, weights,
                quartiles=quartiles,
                weights_for_gas=weights_for_gas,
                rng=rng
            )
            result['scenario'] = scenario_name
            results.append(result)
    
    return pd.DataFrame(results)


# ОСНОВНОЙ БЛОК

if __name__ == '__main__':
    sys.path.append(str(PROJECT_ROOT / 'src'))
   
    print(" ! Запуск сценариев ! ")
    
    # 1. Генерация данных
    employees = generate_employees(n=100, seed=42)
    modules = generate_modules(n=50, seed=42)
    
    # 2. Веса
    weights_path = MODELS_DIR / 'weights.json'
    if weights_path.exists():
        weights_dict = load_weights(weights_path)
        weights = weights_dict['hybrid']
        print(f"Веса загружены из {weights_path}")
    else:
        weights_dict = calculate_all_weights(modules, method='hybrid')
        weights = weights_dict['hybrid']
        print(f"Веса рассчитаны")
    
    # 3. Квартили
    quartiles = None
    quartiles_path = MODELS_DIR / 'quartiles.json'
    if quartiles_path.exists():
        q25, q50, q75 = load_quartiles(quartiles_path)
        quartiles = (q25, q50, q75)
        print(f"Квартили: q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}")
    else:
        print(f"Квартили не найдены")
    
    # 4. Запуск сценариев
    results = run_all_scenarios(
        employees, modules, weights,
        quartiles=quartiles, weights_for_gas=weights, seed=42
    )
    
    # 5. Сводка

    print(" ! РЕЗУЛЬТАТЫ ! ")
  
    summary = results.groupby('scenario').agg(
        mean_GAS=('GAS', 'mean'),
        std_GAS=('GAS', 'std'),
        mean_CS=('CS', 'mean'),
        std_CS=('CS', 'std'),
        mean_ME=('ME', 'mean'),
        mean_TE=('TE', 'mean'),
        mean_time=('time_used', 'mean'),
        mean_modules=('n_modules', 'mean'),
    ).round(4)
    
    print(summary)
    
    # 6. Проверка H1 по GAS и CS
    print("\n--- H1: t-тест S4 vs базовые (GAS) ---")
    s4_gas = results[results['scenario'] == 'S4_VIKOR']['GAS'].values
    
    for baseline in ['S1_Random', 'S2_GreedyTime', 'S3_GreedyGain',
                     'S5_TOPSIS', 'S6_WSA']:
        baseline_gas = results[results['scenario'] == baseline]['GAS'].values
        t_stat, p_value = t_test(s4_gas, baseline_gas)
        p_one = p_value / 2 if t_stat > 0 else 1.0
        sig = "значимо" if (p_one < 0.05 and np.mean(s4_gas) > np.mean(baseline_gas)) else "НЕ значимо"
        print(f"S4 vs {baseline}: t = {t_stat:.3f}, p = {p_one:.6f} ({sig})")
    
    print("\n--- H1: t-тест S4 vs базовые (CS) ---")
    s4_cs = results[results['scenario'] == 'S4_VIKOR']['CS'].values
    
    for baseline in ['S1_Random', 'S2_GreedyTime', 'S3_GreedyGain',
                     'S5_TOPSIS', 'S6_WSA']:
        baseline_cs = results[results['scenario'] == baseline]['CS'].values
        t_stat, p_value = t_test(s4_cs, baseline_cs)
        p_one = p_value / 2 if t_stat > 0 else 1.0
        sig = "значимо" if (p_one < 0.05 and np.mean(s4_cs) > np.mean(baseline_cs)) else "НЕ значимо"
        print(f"S4 vs {baseline}: t = {t_stat:.3f}, p = {p_one:.6f} ({sig})")
    
    print("\nЭксперимент завершён")