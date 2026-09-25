"""
Модуль ранжирования учебных модулей

Реализует:
- VIKOR - основной метод (формулы 19-25 из раздела 2.3)
- TOPSIS - альтернативный метод (для S5)
- WSA - взвешенная сумма (для S6)
- Random - случайный выбор (для S1)
- Greedy by Time - минимальное время (для S2)
- Greedy by Gain - максимальный прирост (для S3)

Использует 8 критериев и типы критериев из weighting.py
"""

import numpy as np

from weighting import CRITERION_NAMES, CRITERIA_TYPES
import sys
from pathlib import Path
    
from data_generation import generate_modules
from weighting import calculate_all_weights


# КОНСТАНТЫ

# Критерии модулей (8 критериев: 5 базовых + 3 компоненты выгорания)
# Импортируются из weighting.py для согласованности

# Типы критериев: 1 - максимизация, -1 - минимизация
# Импортируются из weighting.py для согласованности


# VIKOR - ОСНОВНОЙ МЕТОД (формулы 19-25)

class VIKOR:
    """
    VIKOR - ранжирование с поиском компромисса
    
    Реализует формулы (19)-(25) из раздела 2.3 работы
    Ориентирован на конфликтующие критерии
    """
    
    def __init__(self, weights, v=0.5):
        """
        Параметры:
        - weights: вектор весов критериев (8 значений)
        - v: параметр стратегии (0.5 - равновесие)
        """
        self.weights = np.array(weights)
        self.v = v
    
    def rank(self, matrix, criteria_types=None):
        """
        Ранжирование альтернатив
        
        Параметры:
        - matrix: n_alternatives x m_criteria (8 столбцов)
        - criteria_types: типы критериев (по умолчанию CRITERIA_TYPES)
        
        Возвращает:
        - ranked_indices: список индексов от лучшего к худшему
        - Q: массив компромиссных индексов
        """
        matrix = np.array(matrix, dtype=float)
        n_alt, n_crit = matrix.shape
        
        if criteria_types is None:
            criteria_types = CRITERIA_TYPES
        
        # Шаг 1. Идеальные и антиидеальные значения (формулы 19-20)
        f_star = np.zeros(n_crit)
        f_minus = np.zeros(n_crit)
        for j in range(n_crit):
            if criteria_types[j] == 1:
                f_star[j] = np.max(matrix[:, j])
                f_minus[j] = np.min(matrix[:, j])
            else:
                f_star[j] = np.min(matrix[:, j])
                f_minus[j] = np.max(matrix[:, j])
        
        # Шаг 2. Индексы S и R (формулы 21-22)
        S = np.zeros(n_alt)
        R = np.zeros(n_alt)
        for i in range(n_alt):
            diff = np.abs(f_star - matrix[i, :]) / (f_star - f_minus + 1e-10)
            weighted_diff = self.weights * diff
            S[i] = np.sum(weighted_diff)
            R[i] = np.max(weighted_diff)
        
        # Шаг 3. Компромиссный индекс Q (формула 23)
        S_star, S_minus = np.min(S), np.max(S)
        R_star, R_minus = np.min(R), np.max(R)
        
        Q = self.v * (S - S_star) / (S_minus - S_star + 1e-10) + \
            (1 - self.v) * (R - R_star) / (R_minus - R_star + 1e-10)
        
        # Шаг 4. Проверка условий C1 и C2 (формулы 24-25)
        ranked_indices = list(np.argsort(Q))
        
        return ranked_indices, Q


# TOPSIS - АЛЬТЕРНАТИВНЫЙ МЕТОД

class TOPSIS:
    """
    TOPSIS — близость к идеальному решению
    
    Используется как альтернативный метод для сценария S5
    """
    
    def __init__(self, weights):
        self.weights = np.array(weights)
    
    def rank(self, matrix, criteria_types=None):
        """Ранжирование альтернатив"""
        matrix = np.array(matrix, dtype=float)
        n_alt, n_crit = matrix.shape
        
        if criteria_types is None:
            criteria_types = CRITERIA_TYPES
        
        # Шаг 1. Нормализация
        norm = np.sqrt(np.sum(matrix**2, axis=0))
        matrix_norm = matrix / (norm + 1e-10)
        
        # Шаг 2. Взвешивание
        matrix_weighted = matrix_norm * self.weights
        
        # Шаг 3. Идеальные и антиидеальные решения
        ideal = np.zeros(n_crit)
        anti_ideal = np.zeros(n_crit)
        for j in range(n_crit):
            if criteria_types[j] == 1:
                ideal[j] = np.max(matrix_weighted[:, j])
                anti_ideal[j] = np.min(matrix_weighted[:, j])
            else:
                ideal[j] = np.min(matrix_weighted[:, j])
                anti_ideal[j] = np.max(matrix_weighted[:, j])
        
        # Шаг 4. Расстояния
        S_plus = np.sqrt(np.sum((matrix_weighted - ideal)**2, axis=1))
        S_minus = np.sqrt(np.sum((matrix_weighted - anti_ideal)**2, axis=1))
        
        # Шаг 5. Относительная близость
        scores = S_minus / (S_plus + S_minus + 1e-10)
        
        ranked_indices = list(np.argsort(-scores))
        return ranked_indices, scores


# WSA — ВЗВЕШЕННАЯ СУММА

class WSA:
    """
    WSA (Weighted Sum Approach) - простейший аддитивный метод
    
    Используется как альтернативный метод для сценария S6
    """
    
    def __init__(self, weights):
        self.weights = np.array(weights)
    
    def rank(self, matrix, criteria_types=None):
        """Ранжирование альтернатив."""
        matrix = np.array(matrix, dtype=float)
        n_alt, n_crit = matrix.shape
        
        if criteria_types is None:
            criteria_types = CRITERIA_TYPES
        
        # Min-Max нормализация
        matrix_norm = np.zeros_like(matrix)
        for j in range(n_crit):
            col = matrix[:, j]
            col_min = np.min(col)
            col_max = np.max(col)
            
            if col_max - col_min < 1e-10:
                matrix_norm[:, j] = 0.0
            else:
                if criteria_types[j] == 1:
                    matrix_norm[:, j] = (col - col_min) / (col_max - col_min)
                else:
                    matrix_norm[:, j] = (col_max - col) / (col_max - col_min)
        
        # Взвешенная сумма
        scores = np.sum(matrix_norm * self.weights, axis=1)
        
        ranked_indices = list(np.argsort(-scores))
        return ranked_indices, scores


# RANDOM - СЛУЧАЙНЫЙ ВЫБОР

def random_selection(available_indices, rng=None):
    """Случайный выбор модуля из доступных"""
    if rng is None:
        rng = np.random.default_rng()
    
    return int(rng.choice(available_indices))


# GREEDY BY TIME

def greedy_by_time(available_indices, modules):
    """Жадный выбор по минимальному времени"""
    times = [modules.iloc[idx]['base_time'] for idx in available_indices]
    return available_indices[int(np.argmin(times))]


# GREEDY BY GAIN

def greedy_by_gain(available_indices, modules, deficits,
                   criterion_names=None):
    """
    Жадный выбор по максимальному приросту (с учётом дефицитов)
    
    Параметры:
    - available_indices: список индексов доступных модулей
    - modules: DataFrame с модулями
    - deficits: вектор текущих дефицитов
    - criterion_names: список имён критериев
    
    Возвращает: индекс выбранного модуля
    """
    if criterion_names is None:
        criterion_names = CRITERION_NAMES
    
    gains = []
    for idx in available_indices:
        module_gains = modules.iloc[idx][criterion_names].values.astype(float)
        utility = np.sum(module_gains * deficits)
        gains.append(utility)
    
    return available_indices[int(np.argmax(gains))]


# ОБЩАЯ ФУНКЦИЯ РАНЖИРОВАНИЯ

def rank_modules(available_indices, modules, weights, method='vikor',
                 criteria_types=None, criterion_names=None, deficits=None,
                 rng=None, v=0.5):
    """
    Единая функция ранжирования модулей
    
    Параметры:
    - available_indices: список доступных модулей
    - modules: DataFrame с модулями
    - weights: вектор весов (8 значений)
    - method: 'vikor', 'topsis', 'wsa', 'random', 'greedy_time', 'greedy_gain'
    - criteria_types: типы критериев (по умолчанию CRITERIA_TYPES)
    - criterion_names: имена критериев (по умолчанию CRITERION_NAMES)
    - deficits: текущие дефициты (для greedy_gain)
    - rng: генератор случайных чисел
    - v: параметр стратегии VIKOR
    
    Возвращает: индекс лучшего модуля
    """
    # Простые методы
    if method == 'random':
        return random_selection(available_indices, rng)
    
    if method == 'greedy_time':
        return greedy_by_time(available_indices, modules)
    
    if method == 'greedy_gain':
        if criterion_names is None:
            criterion_names = CRITERION_NAMES
        return greedy_by_gain(available_indices, modules, deficits, criterion_names)
    
    # MCDM-методы
    if criterion_names is None:
        criterion_names = CRITERION_NAMES
    
    if criteria_types is None:
        criteria_types = CRITERIA_TYPES
    
    matrix = modules.iloc[available_indices][criterion_names].values.astype(float)
    
    if method == 'vikor':
        ranker = VIKOR(weights, v=v)
        ranked, _ = ranker.rank(matrix, criteria_types)
    elif method == 'topsis':
        ranker = TOPSIS(weights)
        ranked, _ = ranker.rank(matrix, criteria_types)
    elif method == 'wsa':
        ranker = WSA(weights)
        ranked, _ = ranker.rank(matrix, criteria_types)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    # ranked содержит индексы внутри matrix → преобразуем к исходным
    return available_indices[ranked[0]]


# ОСНОВНОЙ БЛОК

if __name__ == '__main__':   
    sys.path.append(str(Path(__file__).resolve().parent))

    # Генерация
    modules = generate_modules(n=50, seed=42)
    weights_dict = calculate_all_weights(modules, method='hybrid')
    weights = weights_dict['hybrid']
    
    # Доступные модули
    available = list(range(20))
    
    print(" ! Тестирование методов ранжирования ! ")
    print(f"Критериев: {len(CRITERION_NAMES)}")
    print(f"Весов: {len(weights)}")
    print(f"Сумма весов: {np.sum(weights):.4f}")
    
    # VIKOR
    idx_vikor = rank_modules(available, modules, weights, method='vikor')
    print(f"\nVIKOR выбрал модуль: {idx_vikor}")
    
    # TOPSIS
    idx_topsis = rank_modules(available, modules, weights, method='topsis')
    print(f"TOPSIS выбрал модуль: {idx_topsis}")
    
    # WSA
    idx_wsa = rank_modules(available, modules, weights, method='wsa')
    print(f"WSA выбрал модуль: {idx_wsa}")
    
    # Random
    idx_random = rank_modules(available, modules, weights, method='random',
                               rng=np.random.default_rng(42))
    print(f"Random выбрал модуль: {idx_random}")
    
    # Greedy by Time
    idx_time = rank_modules(available, modules, weights, method='greedy_time')
    print(f"Greedy by Time выбрал модуль: {idx_time}")
    
    # Greedy by Gain
    deficits = np.array([0.3, 0.4, 0.2, 0.5, 0.3, 0.4, 0.3, 0.2])
    idx_gain = rank_modules(available, modules, weights, method='greedy_gain',
                             deficits=deficits)
    print(f"Greedy by Gain выбрал модуль: {idx_gain}")
    
    print("\n--- Проверка ---")
    all_indices = [idx_vikor, idx_topsis, idx_wsa, idx_random, idx_time, idx_gain]
    print(f"Все индексы в available: {all(idx in available for idx in all_indices)}")