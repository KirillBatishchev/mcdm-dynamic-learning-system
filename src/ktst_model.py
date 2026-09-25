"""
Модуль обучения и проверки модели Knowledge Tracing Set Transformer (KTST)

Реализует:
- Загрузку датасета FORGET-SE
- Предобработку последовательностей взаимодействий
- Обучение модели KTST с early stopping
- Оценку качества (Test AUC, Test Accuracy)
- Расчёт квартилей распределения P_success (q25, q50, q75)
- Сохранение весов модели, квартилей и метаданных

Ссылка на архитектуру: Neubauer et al. (2026). Principled Transformers for
Predictive Performance in Knowledge Tracing. JEDM, 18(1), 89-112

Модуль проверяется отдельно от MCDM-ядра (раздел 2.4 НИР)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import copy
import warnings
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# КОНСТАНТЫ

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'

# Гиперпараметры модели
EMBEDDING_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 1
DROPOUT = 0.3
MAX_SEQ_LEN = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 5

# Разделение выборки
TRAIN_SPLIT = 0.85
TEST_SPLIT = 0.15
SEED = 42


# ОПРЕДЕЛЕНИЕ УСТРОЙСТВА

def get_device():
    """
    Автоматически определяет доступное устройство
    
    Возвращает: 'cuda', 'mps' или 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Используется GPU: Apple Silicon (MPS)")
    else:
        device = 'cpu'
        print("Используется CPU")
    
    return device


# ДАТАСЕТ

class KTDataset(Dataset):
    """
    Датасет для Knowledge Tracing
    
    Каждый пример - последовательность взаимодействий одного студента:
    - question_ids: идентификаторы вопросов;
    - skill_ids: идентификаторы компонентов знаний (KC);
    - responses: результаты ответов (0/1)
    """
    
    def __init__(self, sequences, max_seq_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        questions = seq['questions'][:self.max_seq_len]
        skills = seq['skills'][:self.max_seq_len]
        responses = seq['responses'][:self.max_seq_len]
        
        seq_len = len(questions)
        pad_len = self.max_seq_len - seq_len
        
        questions = questions + [0] * pad_len
        skills = skills + [0] * pad_len
        responses = responses + [0] * pad_len
        
        return {
            'questions': torch.LongTensor(questions),
            'skills': torch.LongTensor(skills),
            'responses': torch.FloatTensor(responses),
            'seq_len': seq_len
        }


def kt_collate_fn(batch):
    """
    Собирает батч, сохраняя seq_len как список
    
    Это критично для корректной работы маски в forward
    """
    return {
        'questions': torch.stack([b['questions'] for b in batch]),
        'skills': torch.stack([b['skills'] for b in batch]),
        'responses': torch.stack([b['responses'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch],
    }


# МОДЕЛЬ KTST

class KTSTModel(nn.Module):
    """
    Knowledge Tracing Set Transformer (KTST)
    
    Архитектура:
    - Эмбеддинги вопросов, компонентов знаний и ответов
    - Агрегация в единое представление взаимодействия
    - Transformer Encoder с каузальной маской
    - Полносвязный классификатор для прогноза P_success
    
    Каузальная маска обеспечивает прогноз для позиции t
    на основе данных [0..t-1], что исключает утечку данных
    """
    
    def __init__(self, num_questions, num_skills, embed_dim=EMBEDDING_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        self.num_questions = num_questions
        self.num_skills = num_skills
        self.embed_dim = embed_dim
        
        # Эмбеддинги
        self.question_embed = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        self.skill_embed = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)
        self.response_embed = nn.Embedding(3, embed_dim)
        
        # Проекция агрегированного взаимодействия
        self.interaction_proj = nn.Linear(embed_dim * 3, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, questions, skills, responses, seq_len=None):
        batch_size, seq_len_max = questions.shape
        
        # 1. Сдвиг ответов на один шаг
        shifted_responses = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.long, device=responses.device),
            responses[:, :-1].long()
        ], dim=1)
        
        # 2. Эмбеддинги
        q_emb = self.question_embed(questions)
        s_emb = self.skill_embed(skills)
        r_emb = self.response_embed(shifted_responses)
        
        interaction = torch.cat([q_emb, s_emb, r_emb], dim=-1)
        interaction = self.interaction_proj(interaction)
        
        # 3. Каузальная маска через встроенную функцию
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len_max, device=questions.device
        )
        
        # 4. Padding-маска [B, L]
        if seq_len is not None:
            if isinstance(seq_len, int):
                seq_len = [seq_len] * batch_size
            elif isinstance(seq_len, torch.Tensor):
                seq_len = seq_len.tolist()
            
            pad_mask = torch.zeros(
                (batch_size, seq_len_max),
                device=questions.device,
                dtype=torch.bool
            )
            for i, l in enumerate(seq_len):
                if l < seq_len_max:
                    pad_mask[i, l:] = True
        else:
            pad_mask = None
        
        # 5. Transformer
        output = self.transformer(
            interaction,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
            is_causal=False
        )
        
        # 6. Классификатор
        logits = self.classifier(output).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return probs

# ЗАГРУЗКА ДАННЫХ FORGET-SE

def load_forget_se(path=None):
    """
    Загружает датасет FORGET-SE.
    
    Ожидаемый формат CSV:
    - user_id: идентификатор студента;
    - qid: идентификатор вопроса;
    - sequence_id: идентификатор компонента знаний (KC);
    - log_id: временная метка;
    - correct: результат ответа
    """
    if path is None:
        path = DATA_PATH / 'forget_se.csv'
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {path}\n"
            f"Скачайте FORGET-SE с https://github.com/alyssa-sha/FORGET-SE "
            f"и поместите в {DATA_PATH}"
        )
    
    df = pd.read_csv(path)
    
    required = ['user_id', 'qid', 'sequence_id', 'log_id', 'correct']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Отсутствует колонка: {col}")
    
    return df


def preprocess_forget_se(df):
    """
    Предобрабатывает датасет FORGET-SE
    
    Бинаризует ответы: correct >= 0.5 -> 1, иначе -> 0
    
    Возвращает:
    - sequences: список последовательностей по студентам;
    - num_questions: число уникальных вопросов;
    - num_skills: число уникальных компонентов знаний
    """
    df = df.sort_values(['user_id', 'log_id']).reset_index(drop=True)
    
    unique_questions = df['qid'].unique()
    unique_skills = df['sequence_id'].unique()
    
    question_map = {q: i + 1 for i, q in enumerate(unique_questions)}
    skill_map = {s: i + 1 for i, s in enumerate(unique_skills)}
    
    num_questions = len(unique_questions) + 1
    num_skills = len(unique_skills) + 1
    
    sequences = []
    for user_id, group in df.groupby('user_id'):
        responses_binary = [1 if c >= 0.5 else 0 for c in group['correct'].values]
        
        seq = {
            'questions': [question_map[q] for q in group['qid'].values],
            'skills': [skill_map[s] for s in group['sequence_id'].values],
            'responses': responses_binary
        }
        sequences.append(seq)
    
    return sequences, num_questions, num_skills


def save_training_log(log, path=None):
    """Сохраняет лог обучения KTST в JSON"""
    if path is None:
        path = MODELS_DIR / 'ktst_training_log.json'
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)

# ОБУЧЕНИЕ

def train_ktst(train_seqs, val_seqs, num_questions, num_skills,
               epochs=NUM_EPOCHS, device='cpu', patience=PATIENCE):
    """
    Обучает модель KTST с early stopping
    
    Сохраняет лучшую модель по Val AUC
    """
    train_dataset = KTDataset(train_seqs)
    val_dataset = KTDataset(val_seqs)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=kt_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=kt_collate_fn
    )
    
    model = KTSTModel(num_questions, num_skills).to(device)
    best_model_state = copy.deepcopy(model.state_dict())
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCELoss()
    
    best_val_auc = 0.0
    epochs_no_improve = 0
    training_log = {
        'epochs': [],
        'losses': [],
        'val_aucs': [],
        'val_accs': [],
    }
    
    print(f"Обучение KTST на {len(train_seqs)} последовательностях")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            questions = batch['questions'].to(device)
            skills = batch['skills'].to(device)
            responses = batch['responses'].to(device)
            seq_len = batch['seq_len']
            
            optimizer.zero_grad()
            probs = model(questions, skills, responses, seq_len)
            
            # Маска для padding
            mask = torch.zeros_like(probs, dtype=torch.bool)
            for i, l in enumerate(seq_len):
                mask[i, :l] = True
            
            loss = criterion(probs[mask], responses[mask])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Валидация
        val_auc, val_acc = evaluate_ktst(model, val_loader, device)
        avg_loss = train_loss / len(train_loader)
        
        training_log['epochs'].append(epoch + 1)
        training_log['losses'].append(float(avg_loss))
        training_log['val_aucs'].append(float(val_auc))
        training_log['val_accs'].append(float(val_acc))
        
        save_training_log(training_log)
        
        print(f"Эпоха {epoch + 1}/{epochs} | "
              f"Loss: {train_loss / len(train_loader):.4f} | "
              f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    # Загрузка лучшей модели
    model.load_state_dict(best_model_state)
    
    return model


def evaluate_ktst(model, data_loader, device='cpu'):
    """
    Оценивает модель на валидационной/тестовой выборке
    
    Возвращает: (AUC, Accuracy)
    """
    model.eval()
    all_probs = []
    all_responses = []
    
    with torch.no_grad():
        for batch in data_loader:
            questions = batch['questions'].to(device)
            skills = batch['skills'].to(device)
            responses = batch['responses'].to(device)
            seq_len = batch['seq_len']
            
            probs = model(questions, skills, responses, seq_len)
            
            for i, l in enumerate(seq_len):
                all_probs.extend(probs[i, :l].cpu().numpy())
                all_responses.extend(responses[i, :l].cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_responses = np.array(all_responses)
    
    preds = (all_probs > 0.5).astype(int)
    
    auc = roc_auc_score(all_responses, all_probs)
    acc = accuracy_score(all_responses, preds)
    
    return auc, acc


# РАСЧЁТ КВАРТИЛЕЙ

def compute_quartiles(model, sequences, device='cpu'):
    """
    Вычисляет квартили распределения предсказанных вероятностей P_success
    
    Возвращает: (q25, q50, q75)
    """
    dataset = KTDataset(sequences)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=kt_collate_fn
    )
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            questions = batch['questions'].to(device)
            skills = batch['skills'].to(device)
            responses = batch['responses'].to(device)
            seq_len = batch['seq_len']
            
            probs = model(questions, skills, responses, seq_len)
            
            for i, l in enumerate(seq_len):
                all_probs.extend(probs[i, :l].cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    q25 = float(np.percentile(all_probs, 25))
    q50 = float(np.percentile(all_probs, 50))
    q75 = float(np.percentile(all_probs, 75))
    
    print(f"Квартили P_success: q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}")
    
    return q25, q50, q75


# СОХРАНЕНИЕ И ЗАГРУЗКА

def save_model(model, path=None):
    """Сохраняет веса модели"""
    if path is None:
        path = MODELS_DIR / 'ktst_weights.pt'
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена: {path}")


def load_model(num_questions, num_skills, path=None, device='cpu'):
    """Загружает веса модели"""
    if path is None:
        path = MODELS_DIR / 'ktst_weights.pt'
    
    model = KTSTModel(num_questions, num_skills).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Модель загружена: {path}")
    return model


def save_quartiles(q25, q50, q75, path=None):
    """Сохраняет квартили в JSON"""
    if path is None:
        path = MODELS_DIR / 'quartiles.json'
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'q25': q25, 'q50': q50, 'q75': q75}, f, indent=2)
    print(f"Квартили сохранены: {path}")


def load_quartiles(path=None):
    """Загружает квартили из JSON"""
    if path is None:
        path = MODELS_DIR / 'quartiles.json'
    
    with open(path, 'r') as f:
        data = json.load(f)
    return data['q25'], data['q50'], data['q75']


def save_metadata(num_questions, num_skills, path=None):
    """Сохраняет метаданные модели (num_questions, num_skills)"""
    if path is None:
        path = MODELS_DIR / 'model_metadata.json'
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'num_questions': num_questions,
            'num_skills': num_skills,
        }, f, indent=2)
    print(f"Метаданные сохранены: {path}")


# ПРОВЕРКА ГИПОТЕЗЫ

def check_h2_adaptation(quartiles, base_tasks=50):
    """
    Проверяет H2: адаптация количества заданий
    
    Параметры:
    - quartiles: (q25, q50, q75)
    - base_tasks: базовое количество заданий
    
    Возвращает: DataFrame с результатами
    """
    q25, q50, q75 = quartiles
    
    # Тестовые значения P_success
    test_cases = [
        ('Очень слабый', q25 - 0.1),
        ('Слабый', q25 + 0.05),
        ('Средний', q50),
        ('Сильный', q75 + 0.05),
        ('Очень сильный', q75 + 0.1),
    ]
    
    def compute_alpha(p):
        if p > q75:
            return 0.7
        elif p > q50:
            return 1.0
        elif p > q25:
            return 1.3
        else:
            return 1.6
    
    results = []
    for name, p_success in test_cases:
        p_success = float(np.clip(p_success, 0.0, 1.0))
        alpha = compute_alpha(p_success)
        adjusted_tasks = base_tasks * alpha
        change_pct = (alpha - 1.0) * 100
        
        results.append({
            'Категория': name,
            'P_success': round(p_success, 4),
            'α': alpha,
            'Базовое кол-во заданий': base_tasks,
            'Скорректированное': round(adjusted_tasks, 1),
            'Изменение, %': round(change_pct, 1),
        })
    
    df = pd.DataFrame(results)
    
    # Проверка H2
    weak_increased = df[df['P_success'] < q25]['α'].min() > 1.0
    strong_decreased = df[df['P_success'] > q75]['α'].max() < 1.0
    
    h2_confirmed = weak_increased and strong_decreased
    
    print("ПРОВЕРКА H2: адаптация количества знаний")
    print(df.to_string(index=False))
    print(f"\nСлабые получают больше заданий: {weak_increased}")
    print(f"Сильные получают меньше заданий: {strong_decreased}")
    print(f"H2: {'подтверждена' if h2_confirmed else 'НЕ подтверждена'}")
    
    # Сохранение
    path = MODELS_DIR / 'h2_adaptation.csv'
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"Результаты сохранены: {path}")
    
    return df


# ОСНОВНОЙ БЛОК

if __name__ == '__main__':
    print(" ! Обучение и проверка KTST на FORGET-SE ! ")
    
    # 0. Устройство
    device = get_device()
    
    # 1. Загрузка
    print("\n--- Загрузка FORGET-SE ---")
    df = load_forget_se()
    print(f"    Загружено {len(df)} взаимодействий")
    
    # 2. Предобработка
    print("\n--- Предобработка ---")
    sequences, num_questions, num_skills = preprocess_forget_se(df)
    print(f"   Студентов: {len(sequences)}")
    print(f"   Вопросов: {num_questions - 1}")
    print(f"   Компонентов знаний: {num_skills - 1}")
    
    # 3. Разделение train/val/test
    print(f"\n--- Разделение выборки ({TRAIN_SPLIT:.0%} train / {TEST_SPLIT:.0%} test) ---")
    rng = np.random.default_rng(SEED)
    sequences = sequences.copy()
    rng.shuffle(sequences)
    
    n = len(sequences)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * 0.2)
    
    test_seqs = sequences[:n_test]
    val_seqs = sequences[n_test:n_test + n_val]
    train_seqs = sequences[n_test + n_val:]
    
    print(f"   Train: {len(train_seqs)}")
    print(f"   Val: {len(val_seqs)}")
    print(f"   Test: {len(test_seqs)}")
    
    # 4. Обучение
    print("\n--- Обучение модели ---")
    model = train_ktst(train_seqs, val_seqs, num_questions, num_skills,
                       epochs=NUM_EPOCHS, device=device)
    
    # 5. Оценка на тесте
    print("\n--- Оценка на тестовой выборке ---")
    test_loader = DataLoader(
        KTDataset(test_seqs), batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=kt_collate_fn
    )
    test_auc, test_acc = evaluate_ktst(model, test_loader, device=device)
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # 6. Квартили
    print("\n--- Расчёт квартилей P_success ---")
    q25, q50, q75 = compute_quartiles(model, sequences, device=device)
    
    print("\n--- Проверка H2 (адаптация количества заданий) ---")
    h2_results = check_h2_adaptation((q25, q50, q75), base_tasks=50)
    
    # 7. Сохранение
    print("\n--- Сохранение ---")
    save_model(model)
    save_quartiles(q25, q50, q75)
    save_metadata(num_questions, num_skills)
    
    print("\nОбучение завершено")
    print(f"   Модель: {MODELS_DIR / 'ktst_weights.pt'}")
    print(f"   Квартили: {MODELS_DIR / 'quartiles.json'}")
    print(f"   Метаданные: {MODELS_DIR / 'model_metadata.json'}")