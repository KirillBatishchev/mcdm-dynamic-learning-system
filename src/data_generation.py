"""
Модуль генерации синтетических данных для системы адаптивного обучения

Генерирует:
- Профили сотрудников (n=1000) с параметрами из таблиц 1-3
- Учебные модули (N=50) с разделением на профессиональные и личные

Все распределения обоснованы в разделах 1.6 и 2.6.2 работы

ВАЖНО: приросты модулей пропорциональны их времени
Короткие модули дают мало прироста, длинные - много
"""

import numpy as np
import pandas as pd
from pathlib import Path


# КОНСТАНТЫ

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Параметры распределений
STRESS_DISTRIBUTION = {
    '<24':   [0.37, 0.18, 0.45],
    '24-33': [0.21, 0.25, 0.54],
    '34-43': [0.26, 0.29, 0.45],
    '44-57': [0.36, 0.27, 0.37],
    '58-77': [0.45, 0.27, 0.28],
    '>77':   [0.69, 0.18, 0.13],
}

# Компоненты выгорания
BURNOUT_PARAMS = {
    'M': {
        'SI':  {'mean': 20.37, 'std': 8.988, 'min': 0, 'max': 54},
        'DP':  {'mean': 10.61, 'std': 5.771, 'min': 0, 'max': 30},
        'RLD': {'mean': 31.04, 'std': 6.573, 'min': 0, 'max': 48},
    },
    'F': {
        'SI':  {'mean': 22.15, 'std': 8.574, 'min': 0, 'max': 54},
        'DP':  {'mean': 9.97,  'std': 5.214, 'min': 0, 'max': 30},
        'RLD': {'mean': 32.18, 'std': 7.196, 'min': 0, 'max': 48},
    }
}

# Проактивные аттитюды по полу
PROACTIVITY_PARAMS = {
    'M': {'mean': 26.2, 'std': 4.1, 'min': 8, 'max': 32},
    'F': {'mean': 25.2, 'std': 3.7, 'min': 8, 'max': 32},
}

# Целевой профиль (эталон для должности "Инженер-технолог")
TARGET_PROFILE = {
    'target_stress': 8.5,
    'target_reflexivity': 160,
    'target_digital': 0.95,
    'target_knowledge': 0.95,
    'target_proactivity': 35,
    'target_IntPV': 2.0,
}

# Диапазон времени модуля (используется для нормализации приростов)
TIME_MIN = 8.0
TIME_MAX = 125.0


# ГЕНЕРАЦИЯ СОТРУДНИКОВ

def generate_stress_resistance(age_groups, rng):
    """Генерация стрессоустойчивости"""
    values = np.zeros(len(age_groups))
    for i, age in enumerate(age_groups):
        SL = rng.choice([3, 6, 8], p=STRESS_DISTRIBUTION[age])
        values[i] = 10 - SL
    return values


def generate_reflexivity(n, rng):
    """Генерация рефлексивности: смесь трёх нормальных распределений"""
    values = np.zeros(n)
    levels = rng.choice(['low', 'medium', 'high'], size=n, p=[0.35, 0.55, 0.10])
    for i, level in enumerate(levels):
        if level == 'low':
            values[i] = rng.normal(98, 10)
        elif level == 'medium':
            values[i] = rng.normal(127, 8)
        else:
            values[i] = rng.normal(155, 10)
    return np.clip(values, 70, 180)


def generate_digital_literacy(n, rng):
    """Генерация цифровой грамотности: 4% / 75% / 21%"""
    values = np.zeros(n)
    for i in range(n):
        category = rng.choice(
            ['начальный', 'средний', 'продвинутый'],
            p=[0.04, 0.75, 0.21]
        )
        if category == 'начальный':
            values[i] = rng.uniform(0.1, 0.3)
        elif category == 'средний':
            values[i] = rng.uniform(0.4, 0.7)
        else:
            values[i] = rng.uniform(0.75, 0.95)
    return values


def generate_burnout(genders, rng):
    """Генерация трёх компонент выгорания и интегрального показателя"""
    n = len(genders)
    SI = np.zeros(n)
    DP = np.zeros(n)
    RLD = np.zeros(n)
    
    for i, gender in enumerate(genders):
        params = BURNOUT_PARAMS[gender]
        SI[i] = rng.normal(params['SI']['mean'], params['SI']['std'])
        DP[i] = rng.normal(params['DP']['mean'], params['DP']['std'])
        RLD[i] = rng.normal(params['RLD']['mean'], params['RLD']['std'])
    
    # Обрезка до границ шкал
    SI = np.clip(SI, 0, 54)
    DP = np.clip(DP, 0, 30)
    RLD = np.clip(RLD, 0, 48)
    
    # Интегральный показатель
    IntPV = 4.386 + 0.1155 * SI + 0.1747 * DP - 0.0998 * RLD
    
    return SI, DP, RLD, IntPV


def generate_employees(n=100, seed=42):
    """
    Генерирует n сотрудников с параметрами из таблиц 1-3
    
    Возвращает: DataFrame с профилями сотрудников
    """
    rng = np.random.default_rng(seed=seed)
    
    # Пол: 50/50
    genders = rng.choice(['M', 'F'], size=n, p=[0.5, 0.5])
    
    # Возрастные группы: равномерно по 6 группам
    age_groups = rng.choice(
        ['<24', '24-33', '34-43', '44-57', '58-77', '>77'],
        size=n
    )
    
    # Стрессоустойчивость
    stress_resistance = generate_stress_resistance(age_groups, rng)
    
    # Рефлексивность
    reflexivity = generate_reflexivity(n, rng)
    
    # Цифровая грамотность
    digital_literacy = generate_digital_literacy(n, rng)
    
    # Начальный уровень знаний с шумом
    knowledge = np.clip(rng.normal(0.5, 0.2, n) + rng.normal(0, 0.05, n), 0, 1)
    
    # Доступное время
    available_time = rng.uniform(16, 250, n)
    
    # Проактивные аттитюды (по полу)
    proactivity = np.zeros(n)
    for i, gender in enumerate(genders):
        p = PROACTIVITY_PARAMS[gender]
        proactivity[i] = rng.normal(p['mean'], p['std'])
    proactivity = np.clip(proactivity, 8, 32)
    
    # Компоненты выгорания
    SI, DP, RLD, IntPV = generate_burnout(genders, rng)

    df = pd.DataFrame({
        'employee_id': range(n),
        'gender': genders,
        'age_group': age_groups,
        'stress_resistance': stress_resistance,
        'reflexivity': reflexivity,
        'digital_literacy': digital_literacy,
        'knowledge': knowledge,
        'proactivity': proactivity,
        'SI': SI,
        'DP': DP,
        'RLD': RLD,
        'IntPV': IntPV,
        'available_time': available_time
    })
    
    # Добавление целевых значений
    for key, value in TARGET_PROFILE.items():
        df[key] = value
    
    return df


# ГЕНЕРАЦИЯ МОДУЛЕЙ

def _time_factor(base_time):
    """
    Нормализованный коэффициент времени модуля
    
    Возвращает значение из [0, 1]:
    - 0 - минимальное время (8 ч);
    - 1 - максимальное время (125 ч)
    """
    return (base_time - TIME_MIN) / (TIME_MAX - TIME_MIN)


def generate_professional_module(module_id, rng):
    """
    Генерация профессионального модуля
    
    Приросты пропорциональны времени модуля:
    короткие модули дают мало прироста, длинные - много
    
    Конфликт критериев: модуль специализирован либо на профессиональных
    компетенциях, либо на цифровой грамотности (но не на обоих одновременно)
    Это соответствует реальности: курс по программированию даёт много
    delta_prof, но мало delta_digital, и наоборот
    """
    base_time = rng.uniform(TIME_MIN, TIME_MAX)
    tf = _time_factor(base_time)   # [0, 1]
    
    # Конфликт: модуль - либо про prof, либо про digital
    if rng.random() < 0.5:
        delta_prof = rng.uniform(0.15, 0.25) * tf
        delta_digital = rng.uniform(0.02, 0.08) * tf
    else:
        delta_prof = rng.uniform(0.05, 0.12) * tf
        delta_digital = rng.uniform(0.12, 0.20) * tf
    
    return {
        'module_id': module_id,
        'type': 'professional',
        'base_time': base_time,
        'workplace_learning': rng.binomial(1, 0.6),
        'xr_format': rng.binomial(1, 0.2),
        'tasks_count': rng.uniform(5, 100),
        'difficulty': rng.uniform(1, 5),
        'delta_prof': delta_prof,
        'delta_digital': delta_digital,
        'delta_stress': 0.0,
        'delta_reflex': 0.0,
        'delta_proact': 0.0,
        'delta_SI': 0.0,
        'delta_DP': 0.0,
        'delta_RLD': 0.0,
    }


def generate_personal_module(module_id, rng, sigma_SI=8.574, sigma_DP=5.214,
                              sigma_RLD=7.196, sigma_proact=4.1):
    """
    Генерация личного модуля
    
    Приросты пропорциональны времени модуля:
    короткие модули дают мало прироста, длинные - много
    
    Личный модуль относится к одной из трёх групп:
    - stress: снижает ΣИ, ДП; повышает стрессоустойчивость
    - reflex: повышает рефлексивность
    - proactive: повышает проактивные аттитюды и РЛД
    """
    group = rng.choice(['stress', 'reflex', 'proactive'])
    
    base_time = rng.uniform(TIME_MIN, TIME_MAX)
    tf = _time_factor(base_time)
    
    module = {
        'module_id': module_id,
        'type': 'personal',
        'base_time': base_time,
        'workplace_learning': rng.binomial(1, 0.6),
        'xr_format': rng.binomial(1, 0.2),
        'tasks_count': rng.uniform(5, 100),
        'difficulty': rng.uniform(1, 5),
        'delta_prof': 0.0,
        'delta_digital': 0.0,
        'delta_stress': 0.0,
        'delta_reflex': 0.0,
        'delta_proact': 0.0,
        'delta_SI': 0.0,
        'delta_DP': 0.0,
        'delta_RLD': 0.0,
    }
    
    if group == 'stress':
        module['delta_stress'] = rng.uniform(0.15, 0.34) * tf
        module['delta_SI'] = -rng.uniform(0.10, 0.25 * sigma_SI) * tf
        module['delta_DP'] = -rng.uniform(0, 0.10 * sigma_DP) * tf 
    elif group == 'reflex':
        module['delta_reflex'] = rng.uniform(0.05, 0.15) * tf
    else:  # proactive
        module['delta_proact'] = rng.uniform(0, 0.30 * sigma_proact) * tf
        module['delta_RLD'] = rng.uniform(0, 0.43 * sigma_RLD) * tf
    
    return module


def generate_modules(n=50, professional_share=0.7, seed=42):
    """
    Генерирует n учебных модулей с разделением на типы
    
    Параметры:
    - n: количество модулей
    - professional_share: доля профессиональных модулей (0.7 = 70%)
    - seed: seed для воспроизводимости
    
    Возвращает: DataFrame с модулями
    """
    rng = np.random.default_rng(seed=seed)
    modules = []
    
    for i in range(n):
        is_professional = rng.random() < professional_share
        if is_professional:
            module = generate_professional_module(i, rng)
        else:
            module = generate_personal_module(i, rng)
        modules.append(module)
    
    return pd.DataFrame(modules)


# ОСНОВНОЙ БЛОК

if __name__ == '__main__':
    # Генерация данных
    employees = generate_employees(n=1000, seed=42)
    modules = generate_modules(n=50, professional_share=0.7, seed=42)
    
    employees_path = PROJECT_ROOT / 'data' / 'employees.csv'
    modules_path = PROJECT_ROOT / 'data' / 'modules.csv'

    (PROJECT_ROOT / 'data').mkdir(parents=True, exist_ok=True)
    
    # Сохранение в CSV
    employees.to_csv(employees_path, index=False)
    modules.to_csv(modules_path, index=False)
    
    print(" ! Данные сформированы ! ")
    print(f"\nСотрудники: {len(employees)}")
    print(f"Модули: {len(modules)}")
    
    print("\n--- Распределение сотрудников ---")
    print(f"M: {(employees['gender'] == 'M').sum()}")
    print(f"F: {(employees['gender'] == 'F').sum()}")
    print(f"Avg стрессоустойчивость: {employees['stress_resistance'].mean():.2f}")
    print(f"Avg рефлексивность: {employees['reflexivity'].mean():.2f}")
    print(f"Avg цифровая грамотность: {employees['digital_literacy'].mean():.2f}")
    print(f"Avg ИнтПВ: {employees['IntPV'].mean():.2f}")
    
    print("\n--- Распределение модулей ---")
    print(modules['type'].value_counts())
    print(f"\Avg время профессионального модуля: "
          f"{modules[modules['type'] == 'professional']['base_time'].mean():.2f} ч")
    print(f"Avg время личного модуля: "
          f"{modules[modules['type'] == 'personal']['base_time'].mean():.2f} ч")
    
    print("\n--- Проверка пропорциональности приростов ---")
    prof = modules[modules['type'] == 'professional']
    
    # Корреляция между временем и приростом
    corr_prof = prof['base_time'].corr(prof['delta_prof'])
    corr_digital = prof['base_time'].corr(prof['delta_digital'])
    print(f"Корреляция base_time - delta_prof: {corr_prof:.4f}")
    print(f"Корреляция base_time - delta_digital: {corr_digital:.4f}")
    
    # Сравнение коротких и длинных модулей
    short = prof[prof['base_time'] < 40]
    long = prof[prof['base_time'] > 90]
    print(f"\nКороткие модули (< 40 ч): {len(short)}, "
          f"avg delta_prof = {short['delta_prof'].mean():.4f}")
    print(f"Длинные модули (> 90 ч): {len(long)}, "
          f"avg delta_prof = {long['delta_prof'].mean():.4f}")
    
    print("\n--- Проверка разделения ---")
    personal = modules[modules['type'] == 'personal']
    
    print(f"Профессиональные модули с delta_prof > 0: "
          f"{(prof['delta_prof'] > 0).sum()} из {len(prof)}")
    print(f"Личные модули с delta_prof == 0: "
          f"{(personal['delta_prof'] == 0).sum()} из {len(personal)}")
    print(f"Личные модули с delta_digital == 0: "
          f"{(personal['delta_digital'] == 0).sum()} из {len(personal)}")
    
    print(f"\nФайлы сохранены:")
    print(f"  {employees_path}")
    print(f"  {modules_path}")