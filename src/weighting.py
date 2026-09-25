"""
Модуль расчёта весов критериев для многокритериальной модели

Реализует:
- FUCOM - субъективное взвешивание (формулы 1-3)
- CRITIC - объективное взвешивание (формулы 4-11)
- MEREC - объективное взвешивание (формулы 13-18)
- Гибридное взвешивание - мультипликативная нормализация (формула 12)
- Сохранение и загрузку весов в JSON.
"""

import numpy as np
import json
from pathlib import Path
import sys
from data_generation import generate_modules


# КОНСТАНТЫ
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# Критерии модулей (8 критериев: 5 базовых + 3 компоненты выгорания)
CRITERION_NAMES = [
    'delta_prof',       # прирост профессиональных компетенций
    'delta_digital',    # прирост цифровой грамотности
    'delta_stress',     # прирост стрессоустойчивости
    'delta_reflex',     # прирост рефлексивности
    'delta_proact',     # прирост проактивных аттитюдов
    'delta_SI',         # снижение эмоционального истощения
    'delta_DP',         # снижение деперсонализации
    'delta_RLD',        # прирост редукции личных достижений
]

# Типы критериев: 1 - максимизация, -1 - минимизация
# Для delta_SI и delta_DP значения отрицательные (снижение), поэтому
# «больше по модулю - лучше» - учитываем как минимизацию.
CRITERIA_TYPES = [1, 1, 1, 1, 1, -1, -1, 1]

# Ранжирование критериев экспертом (по убыванию важности)
# Обоснование: профессиональные компетенции важнее для производства,
# цифровая грамотность - критична для Индустрии 4.0/5.0,
# снижение истощения - критично для работоспособности,
# остальные характеристики - поддерживающие.
EXPERT_RANKINGS = [
    'delta_prof',
    'delta_digital',
    'delta_SI',
    'delta_stress',
    'delta_DP',
    'delta_RLD',
    'delta_reflex',
    'delta_proact',
]

# Сравнительные приоритеты между соседними критериями (n-1 = 7 значений)
# φ_1/2 = 1.5 (prof важнее digital в 1.5 раза)
# φ_2/3 = 1.3 (digital важнее SI в 1.3 раза)
# φ_3/4 = 1.1 (SI важнее stress в 1.1 раза)
# φ_4/5 = 1.0 (stress и DP равны)
# φ_5/6 = 1.0 (DP и RLD равны)
# φ_6/7 = 1.1 (RLD важнее reflex в 1.1 раза)
# φ_7/8 = 1.0 (reflex и proact равны)
EXPERT_PRIORITIES = [1.5, 1.3, 1.1, 1.0, 1.0, 1.1, 1.0]


# FUCOM - СУБЪЕКТИВНОЕ ВЗВЕШИВАНИЕ

class FUCOM:
    """
    Full Consistency Method (FUCOM) - субъективное взвешивание
    
    Реализует формулы (1)-(3) из раздела 2.2 работы
    Требует всего n-1 сравнений, что снижает нагрузку на эксперта
    """
    
    def __init__(self, rankings, priorities):
        """
        Параметры:
        - rankings: список критериев в порядке убывания важности
        - priorities: сравнительные приоритеты φ между соседними критериями (n-1 значений)
        """
        self.rankings = rankings
        self.priorities = priorities
        self.n = len(rankings)
        
        if len(priorities) != self.n - 1:
            raise ValueError(
                f"Ожидалось {self.n - 1} приоритетов, получено {len(priorities)}"
            )
    
    def calculate(self):
        """
        Расчёт весов методом FUCOM
        
        Возвращает:
        - weights (np.ndarray): вектор весов (сумма = 1)
        - dfc (float): показатель согласованности χ (0 = идеально)
        """
        # Аналитический подход: веса обратно пропорциональны рангам
        # с учётом приоритетов. Полная оптимизационная модель (3) -
        # направление развития
        weights = np.ones(self.n)
        for i in range(1, self.n):
            weights[i] = weights[i-1] / self.priorities[i-1]
        
        # Нормализация
        weights = weights / np.sum(weights)
        
        # Проверка согласованности
        dfc = 0.0
        for i in range(self.n - 1):
            expected_ratio = self.priorities[i]
            actual_ratio = weights[i] / weights[i+1]
            dfc = max(dfc, abs(actual_ratio - expected_ratio))
        
        dfc = dfc / max(self.priorities) if max(self.priorities) > 0 else 0.0
        
        return weights, dfc


# CRITIC - ОБЪЕКТИВНОЕ ВЗВЕШИВАНИЕ

class CRITIC:
    """
    CRITIC - объективное взвешивание через контрастность и конфликт
    
    Реализует формулы (4)-(11) из раздела 2.2 работы
    Учитывает типы критериев (максимизация/минимизация)
    """
    
    def __init__(self, decision_matrix, criteria_types=None):
        """
        Параметры:
        - decision_matrix: матрица n_alternatives x m_criteria (np.ndarray)
        - criteria_types: список типов критериев (1 - макс, -1 — мин)
        """
        self.matrix = np.array(decision_matrix, dtype=float)
        self.n, self.m = self.matrix.shape
        self.criteria_types = criteria_types or [1] * self.m
    
    def _normalize(self):
        """Min-Max нормализация (формулы 4–5) с учётом типов"""
        normalized = np.zeros_like(self.matrix)
        for j in range(self.m):
            col = self.matrix[:, j]
            col_min = np.min(col)
            col_max = np.max(col)
            
            if col_max - col_min < 1e-10:
                normalized[:, j] = 0.0
            else:
                if self.criteria_types[j] == 1:
                    # Максимизация: больше — лучше
                    normalized[:, j] = (col - col_min) / (col_max - col_min)
                else:
                    # Минимизация: меньше — лучше
                    normalized[:, j] = (col_max - col) / (col_max - col_min)
        
        return normalized
    
    def _compute_contrast(self, normalized_matrix):
        """Контрастность через стандартное отклонение (формулы 6-7)"""
        return np.std(normalized_matrix, axis=0, ddof=1)
    
    def _compute_conflict(self, normalized_matrix):
        """Конфликт через корреляцию (формулы 8-9)"""
        corr_matrix = np.corrcoef(normalized_matrix, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        conflict = np.zeros(self.m)
        for j in range(self.m):
            conflict[j] = np.sum(1 - np.abs(corr_matrix[j, :]))
        
        return conflict
    
    def calculate(self):
        """Расчёт весов методом CRITIC (формулы 10-11)"""
        normalized = self._normalize()
        contrast = self._compute_contrast(normalized)
        conflict = self._compute_conflict(normalized)
        
        info_measure = contrast * conflict
        
        total = np.sum(info_measure)
        if total < 1e-10:
            weights = np.ones(self.m) / self.m
        else:
            weights = info_measure / total
        
        return weights


# MEREC - ОБЪЕКТИВНОЕ ВЗВЕШИВАНИЕ

class MEREC:
    """
    MEREC - объективное взвешивание через эффект удаления критерия
    
    Реализует формулы (13)-(18) из раздела 2.2 работы
    Учитывает типы критериев (максимизация/минимизация)
    """
    
    def __init__(self, decision_matrix, criteria_types=None):
        self.matrix = np.array(decision_matrix, dtype=float)
        self.n, self.m = self.matrix.shape
        self.criteria_types = criteria_types or [1] * self.m
    
    def _normalize(self):
        """Линейная нормализация (формулы 13-14) с учётом типов"""
        normalized = np.zeros_like(self.matrix)
        for j in range(self.m):
            col = self.matrix[:, j]
            col_max = np.max(col)
            col_min = np.min(col)
            
            if self.criteria_types[j] == 1:
                # Максимизация
                if col_max < 1e-10:
                    normalized[:, j] = 0.0
                else:
                    normalized[:, j] = col / col_max
            else:
                # Минимизация
                if np.abs(col_min) < 1e-10:
                    normalized[:, j] = 0.0
                else:
                    normalized[:, j] = col_min / (col + 1e-10)
        
        normalized = np.maximum(normalized, 1e-10)
        return normalized
    
    def _compute_overall_performance(self, normalized):
        """Общая эффективность S_i (формула 15)"""
        S = np.zeros(self.n)
        for i in range(self.n):
            log_sum = np.sum(np.abs(np.log(normalized[i, :])))
            S[i] = np.log(1 + (1 / self.m) * log_sum)
        return S
    
    def _compute_partial_performance(self, normalized):
        """Частичная эффективность S'_ij (формула 16)"""
        S_prime = np.zeros((self.n, self.m))
        for j in range(self.m):
            for i in range(self.n):
                other_indices = [k for k in range(self.m) if k != j]
                log_sum = np.sum(np.abs(np.log(normalized[i, other_indices])))
                S_prime[i, j] = np.log(1 + (1 / (self.m - 1)) * log_sum)
        return S_prime
    
    def calculate(self):
        """Расчёт весов методом MEREC (формулы 17–18)"""
        normalized = self._normalize()
        S = self._compute_overall_performance(normalized)
        S_prime = self._compute_partial_performance(normalized)
        
        E = np.zeros(self.m)
        for j in range(self.m):
            E[j] = np.sum(np.abs(S_prime[:, j] - S))
        
        total = np.sum(E)
        if total < 1e-10:
            weights = np.ones(self.m) / self.m
        else:
            weights = E / total
        
        return weights


# ГИБРИДНОЕ ВЗВЕШИВАНИЕ

def hybrid_weights(w_subj, w_obj):
    """Гибридное взвешивание - мультипликативная нормализация (формула 12)"""
    w_subj = np.array(w_subj)
    w_obj = np.array(w_obj)
    
    product = w_subj * w_obj
    total = np.sum(product)
    
    if total < 1e-10:
        return np.ones(len(w_subj)) / len(w_subj)
    
    return product / total


# УДОБНАЯ ОБЁРТКА

def calculate_all_weights(modules, method='hybrid'):
    """
    Расчёт всех весов за один вызов
    
    Параметры:
    - modules: DataFrame с модулями (должен содержать CRITERION_NAMES)
    - method: 'hybrid', 'subjective', 'objective', 'critic', 'merec'
    
    Возвращает: dict с весами и именами критериев
    """
    matrix = modules[CRITERION_NAMES].values
    
    # Субъективные веса (FUCOM)
    fucom = FUCOM(EXPERT_RANKINGS, EXPERT_PRIORITIES)
    w_subj_ranked, dfc = fucom.calculate()
    
    ranking_to_index = {name: i for i, name in enumerate(EXPERT_RANKINGS)}
    criterion_to_rank_idx = [ranking_to_index[name] for name in CRITERION_NAMES]
    w_subj = np.array([w_subj_ranked[i] for i in criterion_to_rank_idx])
    
    # Объективные веса с учётом типов критериев
    critic = CRITIC(matrix, criteria_types=CRITERIA_TYPES)
    w_critic = critic.calculate()
    
    merec = MEREC(matrix, criteria_types=CRITERIA_TYPES)
    w_merec = merec.calculate()
    
    # Выбор объективного метода
    if method == 'critic':
        w_obj = w_critic
    elif method == 'merec':
        w_obj = w_merec
    else:
        w_obj = w_critic   # по умолчанию CRITIC
    
    # Гибридные веса
    w_hybrid = hybrid_weights(w_subj, w_obj)
    
    return {
        'criterion_names': CRITERION_NAMES,
        'criteria_types': CRITERIA_TYPES,
        'subjective': w_subj,
        'critic': w_critic,
        'merec': w_merec,
        'objective': w_obj,
        'hybrid': w_hybrid,
        'dfc': dfc,
    }


# СОХРАНЕНИЕ И ЗАГРУЗКА

def _to_list(value):
    """Преобразует numpy array или list в list для JSON"""
    if hasattr(value, 'tolist'):
        return value.tolist()
    return list(value)


def save_weights(weights_dict, path=None):
    """Сохраняет веса в JSON"""
    if path is None:
        path = MODELS_DIR / 'weights.json'
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'criterion_names': weights_dict['criterion_names'],
        'criteria_types': weights_dict.get('criteria_types', [1] * len(weights_dict['criterion_names'])),
        'subjective': _to_list(weights_dict['subjective']),
        'critic': _to_list(weights_dict['critic']),
        'merec': _to_list(weights_dict['merec']),
        'objective': _to_list(weights_dict['objective']),
        'hybrid': _to_list(weights_dict['hybrid']),
        'dfc': float(weights_dict['dfc']),
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Веса сохранены: {path}")


def load_weights(path=None):
    """Загружает веса из JSON"""
    if path is None:
        path = MODELS_DIR / 'weights.json'
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return {
        'criterion_names': data['criterion_names'],
        'criteria_types': data.get('criteria_types', [1] * len(data['criterion_names'])),
        'subjective': np.array(data['subjective']),
        'critic': np.array(data['critic']),
        'merec': np.array(data['merec']),
        'objective': np.array(data['objective']),
        'hybrid': np.array(data['hybrid']),
        'dfc': data['dfc'],
    }


# ОСНОВНОЙ БЛОК

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).resolve().parent))

    
    # Генерация модулей
    modules = generate_modules(n=50, seed=42)
    
    # Расчёт весов
    weights = calculate_all_weights(modules, method='hybrid')
    
    # Вывод
    print("Расчёт весов критериев")
    
    print("\n--- Критерии ---")
    for name, ctype in zip(weights['criterion_names'], weights['criteria_types']):
        direction = "макс." if ctype == 1 else "мин."
        print(f"  {name} ({direction})")
    
    print("\n--- Субъективные веса (FUCOM) ---")
    for name, w in zip(weights['criterion_names'], weights['subjective']):
        print(f"  {name}: {w:.4f}")
    print(f"  DFC (χ): {weights['dfc']:.4f}")
    
    print("\n--- Объективные веса (CRITIC) ---")
    for name, w in zip(weights['criterion_names'], weights['critic']):
        print(f"  {name}: {w:.4f}")
    
    print("\n--- Объективные веса (MEREC) ---")
    for name, w in zip(weights['criterion_names'], weights['merec']):
        print(f"  {name}: {w:.4f}")
    
    print("\n--- Гибридные веса ---")
    for name, w in zip(weights['criterion_names'], weights['hybrid']):
        print(f"  {name}: {w:.4f}")
    
    # Проверка
    print("\n--- Проверка ---")
    print(f"Сумма субъективных: {np.sum(weights['subjective']):.6f}")
    print(f"Сумма объективных: {np.sum(weights['critic']):.6f}")
    print(f"Сумма гибридных: {np.sum(weights['hybrid']):.6f}")
    print(f"Все веса ≥ 0: {np.all(weights['hybrid'] >= 0)}")
    
    # Сохранение
    save_weights(weights)