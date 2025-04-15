import torch
import torch.nn as nn
import torch.nn.functional as F
from samari import LearnableKalmanTracker

class AttentionKalmanTracker(LearnableKalmanTracker):
    """
    Расширение LearnableKalmanTracker с механизмом внимания
    для более эффективного учета предыдущих измерений
    """
    
    def __init__(
        self,
        state_dim=8,
        meas_dim=4,
        process_noise_scale=0.01,
        measurement_noise_scale=0.1,
        dropout_rate=0.1,
        use_adaptive_noise=True,
        attention_dim=32,
        num_heads=4,
    ):
        super().__init__(
            state_dim=state_dim,
            meas_dim=meas_dim,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale,
            dropout_rate=dropout_rate,
            use_adaptive_noise=use_adaptive_noise,
        )
        
        # Параметры внимания
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Проекции для механизма внимания
        self.query_proj = nn.Linear(meas_dim, attention_dim)
        self.key_proj = nn.Linear(meas_dim, attention_dim)
        self.value_proj = nn.Linear(meas_dim, attention_dim)
        
        # Многоголовое внимание
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Выходная проекция
        self.output_proj = nn.Sequential(
            nn.Linear(attention_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Улучшенный estimator с механизмом внимания
        self.confidence_estimator = nn.Sequential(
            nn.Linear(meas_dim * 2 + state_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # История измерений для внимания
        self.measurement_history = []
        
    def reset(self):
        """Сбрасывает состояние трекера"""
        super().reset()
        self.measurement_history = []
        
    def apply_attention(self, current_measurement):
        """Применяет механизм внимания к истории измерений"""
        if len(self.measurement_history) < 2:
            return None
            
        # Подготовка данных для внимания
        history_tensor = torch.cat(self.measurement_history, dim=0).unsqueeze(0)
        current_tensor = current_measurement.unsqueeze(0)
        
        # Проекции
        query = self.query_proj(current_tensor)
        keys = self.key_proj(history_tensor)
        values = self.value_proj(history_tensor)
        
        # Применение внимания
        attn_output, _ = self.multihead_attn(query, keys, values)
        
        # Проекция выхода
        return self.output_proj(attn_output.squeeze(0))
        
    def update(self, measurement):
        """
        Обновляет состояние с использованием механизма внимания
        """
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.tensor(measurement, device=self.F_diag.device).float()
        if len(measurement.shape) == 1:
            measurement = measurement.unsqueeze(0)
            
        # Сохраняем измерение в истории
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > 30:  # Ограничиваем длину истории
            self.measurement_history.pop(0)
            
        # Применяем внимание
        attention_output = self.apply_attention(measurement)
        
        # Стандартное обновление Калмана с добавлением внимания
        if self.mean is None:
            # Инициализация состояния при первом измерении
            batch_size = measurement.shape[0]
            self.mean = torch.zeros(batch_size, self.state_dim, device=measurement.device)
            self.mean[:, :self.meas_dim] = measurement
            
            self.covariance = torch.eye(self.state_dim, device=measurement.device).expand(batch_size, -1, -1)
            self.last_measurement = measurement
            
            return measurement[0]
        
        # Предсказание
        transition_matrix = self._build_F()
        predicted_mean = torch.bmm(transition_matrix, self.mean.unsqueeze(-1)).squeeze(-1)
        
        # Если есть выход механизма внимания, используем его для улучшения предсказания
        if attention_output is not None:
            predicted_mean = predicted_mean + 0.3 * attention_output
            
        Ft = transition_matrix.transpose(1, 2)
        Q = self._build_Q()
        predicted_cov = torch.bmm(torch.bmm(transition_matrix, self.covariance), Ft) + Q
        
        # Инновация
        H = self._build_H()
        Ht = H.transpose(1, 2)
        R = self._build_R(measurement)
        
        innovation = measurement - torch.bmm(H, predicted_mean.unsqueeze(-1)).squeeze(-1)
        
        # Коэффициент Калмана с учетом уверенности
        S = torch.bmm(torch.bmm(H, predicted_cov), Ht) + R
        K = torch.bmm(torch.bmm(predicted_cov, Ht), torch.inverse(S))
        
        # Внимание-улучшенная оценка уверенности
        measurement_weight = torch.sigmoid(self.measurement_weight)
        if self.last_measurement is not None and attention_output is not None:
            input_features = torch.cat([
                innovation, 
                measurement, 
                attention_output
            ], dim=-1)
            confidence = self.confidence_estimator(input_features)
            measurement_weight = confidence * measurement_weight
        
        # Обновление состояния
        self.mean = predicted_mean + measurement_weight * torch.bmm(
            K, innovation.unsqueeze(-1)
        ).squeeze(-1)
        I = torch.eye(self.state_dim, device=K.device).expand_as(predicted_cov)
        self.covariance = torch.bmm((I - torch.bmm(K, H)), predicted_cov)
        
        # Сохраняем текущее измерение
        self.last_measurement = measurement
        
        # Возвращаем обновленный бокс
        return self.mean[0, :self.meas_dim]


class HybridAttentionLSTMTracker(LearnableKalmanTracker):
    """
    Гибридная модель, объединяющая преимущества LSTM и механизма внимания
    с обучаемым фильтром Калмана
    """
    
    def __init__(
        self,
        state_dim=8,
        meas_dim=4,
        process_noise_scale=0.01,
        measurement_noise_scale=0.1,
        dropout_rate=0.1,
        use_adaptive_noise=True,
        lstm_hidden_dim=32,
        lstm_layers=1,
        attention_dim=32,
        num_heads=4,
    ):
        super().__init__(
            state_dim=state_dim,
            meas_dim=meas_dim,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale,
            dropout_rate=dropout_rate,
            use_adaptive_noise=use_adaptive_noise,
        )
        
        # Проекция входных данных
        self.input_proj = nn.Linear(meas_dim, lstm_hidden_dim)
        
        # LSTM компоненты - исправлено input_size на lstm_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,  # Исправлено с meas_dim на lstm_hidden_dim
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
        )
        
        # Attention компоненты
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Проекции - остальные части класса остаются без изменений
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Обновленный адаптивный шум
        if use_adaptive_noise:
            self.adaptive_noise = nn.Sequential(
                nn.Linear(meas_dim * 2 + lstm_hidden_dim * 2, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, state_dim),
                nn.Sigmoid(),
            )
            
        # Улучшенный confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(meas_dim * 2 + lstm_hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # История и состояния
        self.measurement_history = []
        self.hidden_state = None
        
    def reset(self):
        """Сбрасывает состояние трекера"""
        super().reset()
        self.measurement_history = []
        self.hidden_state = None
        
    def update(self, measurement):
        """
        Обновляет состояние с использованием гибридной архитектуры LSTM + Attention
        """
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.tensor(measurement, device=self.F_diag.device).float()
        if len(measurement.shape) == 1:
            measurement = measurement.unsqueeze(0)
            
        # Сохраняем измерение в истории
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > 50:
            self.measurement_history.pop(0)
            
        # Обработка с помощью LSTM и Attention
        lstm_output = None
        attn_output = None
        
        if len(self.measurement_history) >= 2:
            # Подготовка данных
            history_tensor = torch.cat(self.measurement_history, dim=0).unsqueeze(0)
            projected_history = self.input_proj(history_tensor)
            
            # LSTM обработка
            if self.hidden_state is None:
                lstm_out, self.hidden_state = self.lstm(projected_history)
            else:
                # Используем только последнее измерение для обновления
                lstm_out, self.hidden_state = self.lstm(
                    projected_history[:, -1:].contiguous(), 
                    (self.hidden_state[0].detach(), self.hidden_state[1].detach())
                )
                
            # Attention обработка
            current_embedded = self.input_proj(measurement).unsqueeze(0)
            attn_output, _ = self.multihead_attn(
                current_embedded,
                lstm_out,
                lstm_out
            )
            
            # Объединяем выходы
            combined = torch.cat([
                lstm_out[:, -1], 
                attn_output.squeeze(0)
            ], dim=-1)
            
            hybrid_output = self.output_proj(combined)
        
        # Стандартное обновление Калмана
        if self.mean is None:
            # Инициализация состояния при первом измерении
            batch_size = measurement.shape[0]
            self.mean = torch.zeros(batch_size, self.state_dim, device=measurement.device)
            self.mean[:, :self.meas_dim] = measurement
            
            self.covariance = torch.eye(self.state_dim, device=measurement.device).expand(batch_size, -1, -1)
            self.last_measurement = measurement
            
            return measurement[0]
        
        # Предсказание
        transition_matrix = self._build_F()
        predicted_mean = torch.bmm(transition_matrix, self.mean.unsqueeze(-1)).squeeze(-1)
        
        # Если есть выход гибридной модели, используем его
        if 'hybrid_output' in locals() and hybrid_output is not None:
            predicted_mean = predicted_mean + 0.3 * hybrid_output
            
        Ft = transition_matrix.transpose(1, 2)
        Q = self._build_Q()
        predicted_cov = torch.bmm(torch.bmm(transition_matrix, self.covariance), Ft) + Q
        
        # Инновация
        H = self._build_H()
        Ht = H.transpose(1, 2)
        R = self._build_R(measurement)
        
        innovation = measurement - torch.bmm(H, predicted_mean.unsqueeze(-1)).squeeze(-1)
        
        # Коэффициент Калмана с учетом уверенности
        S = torch.bmm(torch.bmm(H, predicted_cov), Ht) + R
        K = torch.bmm(torch.bmm(predicted_cov, Ht), torch.inverse(S))
        
        # Расчет уверенности в измерении
        measurement_weight = torch.sigmoid(self.measurement_weight)
        if self.last_measurement is not None and 'combined' in locals() and combined is not None:
            input_features = torch.cat([
                innovation, 
                measurement, 
                combined
            ], dim=-1)
            confidence = self.confidence_estimator(input_features)
            measurement_weight = confidence * measurement_weight
        
        # Обновление состояния
        self.mean = predicted_mean + measurement_weight * torch.bmm(
            K, innovation.unsqueeze(-1)
        ).squeeze(-1)
        I = torch.eye(self.state_dim, device=K.device).expand_as(predicted_cov)
        self.covariance = torch.bmm((I - torch.bmm(K, H)), predicted_cov)
        
        # Сохраняем текущее измерение
        self.last_measurement = measurement
        
        # Возвращаем обновленный бокс
        return self.mean[0, :self.meas_dim]
    

class MABMAKalmanTracker(LearnableKalmanTracker):
    """
    Реализация Multi-Agent Behavior Modeling and Analysis (MABMA) архитектуры
    для трекинга объектов. Использует несколько специализированных агентов для моделирования
    разных аспектов движения.
    """
    
    def __init__(
        self,
        state_dim=8,
        meas_dim=4,
        process_noise_scale=0.01,
        measurement_noise_scale=0.1,
        dropout_rate=0.1,
        use_adaptive_noise=True,
        hidden_dim=32,
        num_agents=4,
        agent_dim=16,
    ):
        super().__init__(
            state_dim=state_dim,
            meas_dim=meas_dim,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale,
            dropout_rate=dropout_rate,
            use_adaptive_noise=use_adaptive_noise,
        )
        
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.agent_dim = agent_dim
        
        # Проекции входных данных для агентов
        self.input_proj = nn.Linear(meas_dim, hidden_dim)
        
        # Создаем несколько специализированных агентов
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, agent_dim),
                nn.LayerNorm(agent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(agent_dim, agent_dim),
                nn.LayerNorm(agent_dim),
                nn.ReLU(),
            ) for _ in range(num_agents)
        ])
        
        # Веса для агентов (обучаемые)
        self.agent_weights = nn.Parameter(torch.ones(num_agents) / num_agents)
        
        # Гейты для учета уверенности каждого агента
        self.agent_confidence = nn.ModuleList([
            nn.Sequential(
                nn.Linear(agent_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_agents)
        ])
        
        # Выходная проекция для объединения результатов агентов
        self.agent_combiner = nn.Sequential(
            nn.Linear(agent_dim * num_agents, state_dim),
            nn.LayerNorm(state_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Временной контекст для агентов
        self.temporal_context = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Контекстный проектор
        self.context_proj = nn.Linear(hidden_dim, agent_dim * num_agents)
        
        # Модуль внимания для взаимодействия между агентами
        self.inter_agent_attention = nn.MultiheadAttention(
            embed_dim=agent_dim,
            num_heads=min(4, agent_dim // 4),
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Улучшенный confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(meas_dim * 2 + agent_dim * num_agents, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # История измерений и состояния
        self.measurement_history = []
        self.hidden_state = None
        self.contexts = []
        
    def reset(self):
        """Сбрасывает состояние трекера"""
        super().reset()
        self.measurement_history = []
        self.hidden_state = None
        self.contexts = []
        
    def update(self, measurement):
        """
        Обновляет состояние с использованием MABMA архитектуры
        """
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.tensor(measurement, device=self.F_diag.device).float()
        if len(measurement.shape) == 1:
            measurement = measurement.unsqueeze(0)
            
        # Сохраняем измерение в истории
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > 30:
            self.measurement_history.pop(0)
            
        # Проекция текущего измерения
        current_feat = self.input_proj(measurement)
        
        # Временной контекст
        context_vector = None
        if len(self.measurement_history) >= 2:
            # Подготовка данных для временного контекста
            history_tensor = torch.cat([self.input_proj(m) for m in self.measurement_history], dim=0).unsqueeze(0)
            
            if self.hidden_state is None:
                context_out, self.hidden_state = self.temporal_context(history_tensor)
            else:
                context_out, self.hidden_state = self.temporal_context(
                    history_tensor[:, -1:],
                    self.hidden_state.detach()
                )
            
            # Сохраняем контекст для текущего шага
            self.contexts.append(context_out[:, -1])
            if len(self.contexts) > 10:
                self.contexts.pop(0)
                
            # Объединяем контекст
            context_vector = self.context_proj(context_out[:, -1])
        
        # Применяем агентов к текущему измерению
        agent_outputs = []
        agent_confidences = []
        
        # Проекция текущего измерения через каждого агента
        for i, agent in enumerate(self.agents):
            agent_out = agent(current_feat)
            agent_outputs.append(agent_out)
            
            # Оценка уверенности агента
            confidence = self.agent_confidence[i](agent_out)
            agent_confidences.append(confidence)
        
        # Взаимодействие между агентами через внимание
        # Исправляем форму тензора для соответствия ожиданиям MultiheadAttention
        # MultiheadAttention ожидает тензор формы [seq_len, batch_size, embed_dim]
        agent_outputs_stacked = torch.stack(agent_outputs)  # [num_agents, batch_size, agent_dim]
        
        # Применяем внимание (seq_len = num_agents, batch_size = 1, embed_dim = agent_dim)
        agent_attn_out, _ = self.inter_agent_attention(
            agent_outputs_stacked,
            agent_outputs_stacked,
            agent_outputs_stacked
        )
        
        # agent_attn_out имеет форму [num_agents, batch_size, agent_dim]
        agent_outputs_updated = agent_attn_out
        
        # Взвешиваем агентов в соответствии с их уверенностью
        weights = F.softmax(self.agent_weights, dim=0)
        weighted_outputs = []
        
        for i in range(self.num_agents):
            # Умножаем выход агента на его вес и уверенность
            weighted_out = agent_outputs_updated[i] * weights[i] * agent_confidences[i]
            weighted_outputs.append(weighted_out)
        
        # Объединяем взвешенные выходы агентов
        agent_combined = torch.cat(weighted_outputs, dim=-1)
        
        # Применяем контекстную информацию если она доступна
        if context_vector is not None:
            # Модулируем выходы агентов с помощью контекста
            context_gating = torch.sigmoid(context_vector)
            agent_combined = agent_combined * context_gating
        
        # Итоговое предсказание на основе агентов
        agent_prediction = self.agent_combiner(agent_combined)
        
        # Стандартное обновление Калмана
        if self.mean is None:
            # Инициализация состояния при первом измерении
            batch_size = measurement.shape[0]
            self.mean = torch.zeros(batch_size, self.state_dim, device=measurement.device)
            self.mean[:, :self.meas_dim] = measurement
            
            self.covariance = torch.eye(self.state_dim, device=measurement.device).expand(batch_size, -1, -1)
            self.last_measurement = measurement
            
            return measurement[0]
        
        # Предсказание
        transition_matrix = self._build_F()
        predicted_mean = torch.bmm(transition_matrix, self.mean.unsqueeze(-1)).squeeze(-1)
        
        # Интегрируем предсказание агентов
        if 'agent_prediction' in locals():
            predicted_mean = predicted_mean + 0.3 * agent_prediction
            
        Ft = transition_matrix.transpose(1, 2)
        Q = self._build_Q()
        predicted_cov = torch.bmm(torch.bmm(transition_matrix, self.covariance), Ft) + Q
        
        # Инновация
        H = self._build_H()
        Ht = H.transpose(1, 2)
        R = self._build_R(measurement)
        
        innovation = measurement - torch.bmm(H, predicted_mean.unsqueeze(-1)).squeeze(-1)
        
        # Коэффициент Калмана с учетом уверенности
        S = torch.bmm(torch.bmm(H, predicted_cov), Ht) + R
        K = torch.bmm(torch.bmm(predicted_cov, Ht), torch.inverse(S))
        
        # Оценка уверенности в измерении на основе агентов
        measurement_weight = torch.sigmoid(self.measurement_weight)
        if self.last_measurement is not None and 'agent_combined' in locals():
            input_features = torch.cat([
                innovation,
                measurement,
                agent_combined
            ], dim=-1)
            confidence = self.confidence_estimator(input_features)
            measurement_weight = confidence * measurement_weight
        
        # Обновление состояния
        self.mean = predicted_mean + measurement_weight * torch.bmm(
            K, innovation.unsqueeze(-1)
        ).squeeze(-1)
        I = torch.eye(self.state_dim, device=K.device).expand_as(predicted_cov)
        self.covariance = torch.bmm((I - torch.bmm(K, H)), predicted_cov)
        
        # Сохраняем текущее измерение
        self.last_measurement = measurement
        
        # Возвращаем обновленный бокс
        return self.mean[0, :self.meas_dim]


class EnsembleKalmanTracker(LearnableKalmanTracker):
    """
    Ансамблевый трекер, объединяющий различные архитектуры для улучшения точности
    """
    
    def __init__(
        self,
        state_dim=8,
        meas_dim=4,
        process_noise_scale=0.01,
        measurement_noise_scale=0.1,
        dropout_rate=0.1,
        use_adaptive_noise=True,
    ):
        # Отключаем вызов reset() при инициализации базового класса
        self._skip_reset = True
        
        # Сначала вызываем родительский конструктор
        super().__init__(
            state_dim=state_dim,
            meas_dim=meas_dim,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale,
            dropout_rate=dropout_rate,
            use_adaptive_noise=use_adaptive_noise,
        )
        
        # Создаем модели в ансамбле ПОСЛЕ вызова родительского конструктора
        self.attention_tracker = AttentionKalmanTracker(
            state_dim=state_dim,
            meas_dim=meas_dim,
            dropout_rate=dropout_rate
        )
        
        self.mabma_tracker = MABMAKalmanTracker(
            state_dim=state_dim,
            meas_dim=meas_dim,
            dropout_rate=dropout_rate
        )
        
        # Веса для комбинирования предсказаний (обучаемые)
        self.model_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Сеть для динамической оценки весов моделей
        self.weight_estimator = nn.Sequential(
            nn.Linear(meas_dim * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
        # Теперь, когда все инициализировано, сбрасываем состояния
        self._skip_reset = False
        self.reset()
        
    def reset(self):
        """Сбрасывает состояние всех трекеров"""
        # Базовый reset
        super().reset()
        
        # Проверяем, нужно ли пропустить сброс трекеров
        if hasattr(self, '_skip_reset') and self._skip_reset:
            return
        
        # Сбрасываем состояние трекеров, если они определены
        if hasattr(self, 'attention_tracker'):
            self.attention_tracker.reset()
        if hasattr(self, 'mabma_tracker'):
            self.mabma_tracker.reset()
        
    def update(self, measurement):
        """
        Обновляет состояние с использованием ансамбля моделей
        """
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.tensor(measurement, device=self.F_diag.device).float()
        if len(measurement.shape) == 1:
            measurement = measurement.unsqueeze(0)
            
        # Получаем предсказания от моделей
        attn_pred = self.attention_tracker.update(measurement)
        mabma_pred = self.mabma_tracker.update(measurement)
        
        if self.mean is None:
            # Инициализация состояния при первом измерении
            batch_size = measurement.shape[0]
            self.mean = torch.zeros(batch_size, self.state_dim, device=measurement.device)
            self.mean[:, :self.meas_dim] = measurement
            
            self.covariance = torch.eye(self.state_dim, device=measurement.device).expand(batch_size, -1, -1)
            self.last_measurement = measurement
            
            return measurement[0]
        
        # Динамическая оценка весов моделей
        if self.last_measurement is not None:
            # Собираем предсказания в один тензор
            all_preds = torch.cat([
                attn_pred.unsqueeze(0),
                mabma_pred.unsqueeze(0)
            ], dim=0)
            
            input_features = torch.cat([
                measurement.repeat(2, 1),
                self.last_measurement.repeat(2, 1),
                all_preds
            ], dim=-1)
            
            # Получаем веса и берем только первую строку тензора
            dynamic_weights = self.weight_estimator(input_features)[0]
        else:
            # Используем базовые веса если нет предыдущего измерения
            dynamic_weights = F.softmax(self.model_weights, dim=0)
        
        # Комбинируем предсказания с весами напрямую, без конвертации в скаляры
        weighted_pred = dynamic_weights[0] * attn_pred + dynamic_weights[1] * mabma_pred
        
        # Обновляем состояние фильтра Калмана
        transition_matrix = self._build_F()
        predicted_mean = torch.bmm(transition_matrix, self.mean.unsqueeze(-1)).squeeze(-1)
        
        # Интегрируем ансамблевое предсказание
        predicted_mean[:, :self.meas_dim] = 0.7 * predicted_mean[:, :self.meas_dim] + 0.3 * weighted_pred.unsqueeze(0)
            
        Ft = transition_matrix.transpose(1, 2)
        Q = self._build_Q()
        predicted_cov = torch.bmm(torch.bmm(transition_matrix, self.covariance), Ft) + Q
        
        # Инновация
        H = self._build_H()
        Ht = H.transpose(1, 2)
        R = self._build_R(measurement)
        
        innovation = measurement - torch.bmm(H, predicted_mean.unsqueeze(-1)).squeeze(-1)
        
        # Коэффициент Калмана
        S = torch.bmm(torch.bmm(H, predicted_cov), Ht) + R
        K = torch.bmm(torch.bmm(predicted_cov, Ht), torch.inverse(S))
        
        # Обновление состояния
        self.mean = predicted_mean + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        I = torch.eye(self.state_dim, device=K.device).expand_as(predicted_cov)
        self.covariance = torch.bmm((I - torch.bmm(K, H)), predicted_cov)
        
        # Сохраняем текущее измерение
        self.last_measurement = measurement
        
        # Возвращаем обновленный бокс
        return self.mean[0, :self.meas_dim]


if __name__ == "__main__":
    BS, SEQ_LEN, DIM = 10, 64, 4
    x = torch.randn(BS, SEQ_LEN, DIM).float()
    model = EnsembleKalmanTracker()
    
    # Корректный способ обработки последовательности
    outputs = []
    for b in range(BS):
        model.reset()  # Сбросить состояние для новой последовательности
        seq_outputs = []
        
        for t in range(SEQ_LEN):
            # Обрабатываем каждое измерение последовательно
            output = model(x[b, t])
            seq_outputs.append(output)
        
        outputs.append(torch.stack(seq_outputs))
    
    # Собираем результаты в батч
    batch_outputs = torch.stack(outputs)
    print(f"Выходная форма: {batch_outputs.shape}")
    print(f"Количество обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")