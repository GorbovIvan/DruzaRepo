import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
from typing import List, Dict, Any
import json

# ==================== АРХИТЕКТУРА НЕЙРОСЕТИ НА NUMPY ====================

class Conv2D:
    """Сверточный слой"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Инициализация весов (Xavier/Glorot)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)
        
        # Для хранения градиентов
        self.d_weights = None
        self.d_bias = None
        self.input = None
    
    def forward(self, x, training=False):
        """
        Прямой проход свертки
        x: форма (batch_size, channels, height, width)
        """
        self.input = x if training else None
        batch_size, channels, height, width = x.shape
        
        # Добавление padding
        if self.padding > 0:
            x_pad = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), 
                               (self.padding, self.padding)), mode='constant')
        else:
            x_pad = x
        
        out_height = height + 2*self.padding - self.kernel_size + 1
        out_width = width + 2*self.padding - self.kernel_size + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Свертка
        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end = i, i + self.kernel_size
                w_start, w_end = j, j + self.kernel_size
                
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(x_slice * self.weights[k, :, :, :], axis=(1,2,3))
                    output[:, k, i, j] += self.bias[k]
        
        return output
    
    def backward(self, grad_output, learning_rate):
        """Обратный проход (для обучения)"""
        batch_size = grad_output.shape[0]
        _, _, out_height, out_width = grad_output.shape
        
        # Padding входных данных
        x_pad = np.pad(self.input, ((0,0), (0,0), (self.padding, self.padding), 
                                   (self.padding, self.padding)), mode='constant')
        
        # Инициализация градиентов
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)
        dx_pad = np.zeros_like(x_pad)
        
        # Вычисление градиентов
        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end = i, i + self.kernel_size
                w_start, w_end = j, j + self.kernel_size
                
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    self.d_weights[k] += np.sum(x_slice * grad_output[:, k:k+1, i:i+1, j:j+1], axis=0)
                    self.d_bias[k] += np.sum(grad_output[:, k, i, j])
                    
                    dx_pad[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k] * grad_output[:, k:k+1, i:i+1, j:j+1]
        
        # Обновление весов
        self.weights -= learning_rate * self.d_weights / batch_size
        self.bias -= learning_rate * self.d_bias / batch_size
        
        # Обрезаем padding для градиента
        if self.padding > 0:
            return dx_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_pad

class MaxPool2D:
    """Слой подвыборки (Max Pooling)"""
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.input = None
    
    def forward(self, x, training=False):
        """
        Прямой проход пулинга
        x: форма (batch_size, channels, height, width)
        """
        self.input = x if training else None
        batch_size, channels, height, width = x.shape
        
        out_height = height // self.pool_size
        out_width = width // self.pool_size
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size
                
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(x_slice, axis=(2,3))
        
        return output
    
    def backward(self, grad_output, learning_rate=None):
        """Обратный проход пулинга"""
        grad_input = np.zeros_like(self.input)
        batch_size, channels, height, width = grad_output.shape
        
        for i in range(height):
            for j in range(width):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size
                
                x_slice = self.input[:, :, h_start:h_end, w_start:w_end]
                max_mask = x_slice == np.max(x_slice, axis=(2,3), keepdims=True)
                grad_input[:, :, h_start:h_end, w_start:w_end] += \
                    max_mask * grad_output[:, :, i:i+1, j:j+1]
        
        return grad_input

class BatchNorm2D:
    """Слой пакетной нормализации"""
    def __init__(self, num_channels, eps=1e-5, momentum=0.9):
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        
        # Обучаемые параметры
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        
        # Бегущая статистика для инференса
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)
        
        # Градиенты
        self.d_gamma = None
        self.d_beta = None
        self.input = None
        self.normalized = None
        self.var = None
    
    def forward(self, x, training=True):
        """
        Прямой проход BatchNorm
        x: форма (batch_size, channels, height, width)
        """
        self.input = x
        
        if training:
            # Вычисление статистик по батчу
            mean = np.mean(x, axis=(0,2,3), keepdims=True)
            var = np.var(x, axis=(0,2,3), keepdims=True)
            
            # Обновление бегущих статистик
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
            
            # Нормализация
            self.normalized = (x - mean) / np.sqrt(var + self.eps)
            output = self.gamma.reshape(1, -1, 1, 1) * self.normalized + self.beta.reshape(1, -1, 1, 1)
        else:
            # Использование бегущих статистик
            normalized = (x - self.running_mean.reshape(1, -1, 1, 1)) / \
                        np.sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)
            output = self.gamma.reshape(1, -1, 1, 1) * normalized + self.beta.reshape(1, -1, 1, 1)
        
        return output
    
    def backward(self, grad_output, learning_rate):
        """Обратный проход BatchNorm"""
        batch_size = grad_output.shape[0]
        
        # Градиенты для gamma и beta
        self.d_gamma = np.sum(grad_output * self.normalized, axis=(0,2,3))
        self.d_beta = np.sum(grad_output, axis=(0,2,3))
        
        # Градиент для входных данных
        grad_normalized = grad_output * self.gamma.reshape(1, -1, 1, 1)
        
        var = np.var(self.input, axis=(0,2,3), keepdims=True)
        mean = np.mean(self.input, axis=(0,2,3), keepdims=True)
        
        grad_var = np.sum(grad_normalized * (self.input - mean) * -0.5 * (var + self.eps)**(-1.5), 
                         axis=(0,2,3), keepdims=True)
        grad_mean = np.sum(grad_normalized * -1 / np.sqrt(var + self.eps), axis=(0,2,3), keepdims=True) + \
                   grad_var * np.mean(-2 * (self.input - mean), axis=(0,2,3), keepdims=True)
        
        grad_input = grad_normalized / np.sqrt(var + self.eps) + \
                    2 * grad_var * (self.input - mean) / (self.input.shape[0] * self.input.shape[2] * self.input.shape[3]) + \
                    grad_mean / (self.input.shape[0] * self.input.shape[2] * self.input.shape[3])
        
        # Обновление параметров
        self.gamma -= learning_rate * self.d_gamma / batch_size
        self.beta -= learning_rate * self.d_beta / batch_size
        
        return grad_input

class Dense:
    """Полносвязный слой"""
    def __init__(self, in_features, out_features):
        # Инициализация Xavier
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(out_features, in_features) * scale
        self.bias = np.zeros(out_features)
        
        self.input = None
        self.d_weights = None
        self.d_bias = None
    
    def forward(self, x, training=False):
        """
        Прямой проход
        x: форма (batch_size, in_features)
        """
        self.input = x if training else None
        return np.dot(x, self.weights.T) + self.bias
    
    def backward(self, grad_output, learning_rate):
        """Обратный проход"""
        batch_size = grad_output.shape[0]
        
        # Градиенты
        self.d_weights = np.dot(grad_output.T, self.input)
        self.d_bias = np.sum(grad_output, axis=0)
        
        # Градиент для входных данных
        grad_input = np.dot(grad_output, self.weights)
        
        # Обновление весов
        self.weights -= learning_rate * self.d_weights / batch_size
        self.bias -= learning_rate * self.d_bias / batch_size
        
        return grad_input

class Dropout:
    """Слой dropout для регуляризации"""
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
    
    def forward(self, x, training=True):
        """
        Прямой проход
        x: форма (batch_size, features)
        """
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad_output, learning_rate=None):
        """Обратный проход"""
        return grad_output * self.mask

class ReLU:
    """Функция активации ReLU"""
    def forward(self, x, training=False):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output, learning_rate=None):
        return grad_output * (self.input > 0)

class Flatten:
    """Слой для выпрямления тензора"""
    def forward(self, x, training=False):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output, learning_rate=None):
        return grad_output.reshape(self.input_shape)

class GlobalAveragePooling2D:
    """Глобальный средний пулинг"""
    def forward(self, x, training=False):
        self.input_shape = x.shape
        return np.mean(x, axis=(2,3))
    
    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, height, width = self.input_shape
        grad_input = grad_output.reshape(batch_size, channels, 1, 1)
        grad_input = np.repeat(grad_input, height, axis=2)
        grad_input = np.repeat(grad_input, width, axis=3)
        return grad_input

# ==================== ПОЛНАЯ МОДЕЛЬ ====================

class NumPyImageClassifier:
    """Сверточная нейросеть на NumPy для классификации изображений"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.layers = []
        self.training = True
        
        # Сверточные слои
        self.conv1 = Conv2D(3, 32, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(32)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)
        
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(64)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)
        
        self.conv3 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2D(128)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D(2)
        
        self.conv4 = Conv2D(128, 256, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2D(256)
        self.relu4 = ReLU()
        self.pool4 = MaxPool2D(2)
        
        # Классификатор
        self.gap = GlobalAveragePooling2D()
        self.dropout1 = Dropout(0.5)
        self.fc1 = Dense(256, 512)
        self.relu5 = ReLU()
        self.dropout2 = Dropout(0.3)
        self.fc2 = Dense(512, 256)
        self.relu6 = ReLU()
        self.fc3 = Dense(256, num_classes)
        
        # Собираем все слои в список для удобства
        self.layers = [
            self.conv1, self.bn1, self.relu1, self.pool1,
            self.conv2, self.bn2, self.relu2, self.pool2,
            self.conv3, self.bn3, self.relu3, self.pool3,
            self.conv4, self.bn4, self.relu4, self.pool4,
            self.gap, self.dropout1, self.fc1, self.relu5,
            self.dropout2, self.fc2, self.relu6, self.fc3
        ]
    
    def forward(self, x, training=True):
        """Прямой проход через все слои"""
        self.training = training
        
        for layer in self.layers:
            x = layer.forward(x, training)
        
        return x
    
    def backward(self, grad_output, learning_rate):
        """Обратный проход через все слои"""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
        
        return grad_output
    
    def save(self, path):
        """Сохранение модели"""
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                params[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias'):
                params[f'layer_{i}_bias'] = layer.bias
            if hasattr(layer, 'gamma'):
                params[f'layer_{i}_gamma'] = layer.gamma
            if hasattr(layer, 'beta'):
                params[f'layer_{i}_beta'] = layer.beta
            if hasattr(layer, 'running_mean'):
                params[f'layer_{i}_running_mean'] = layer.running_mean
            if hasattr(layer, 'running_var'):
                params[f'layer_{i}_running_var'] = layer.running_var
        
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        print(f"✓ Модель сохранена в {path}")
    
    def load(self, path):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and f'layer_{i}_weights' in params:
                layer.weights = params[f'layer_{i}_weights']
            if hasattr(layer, 'bias') and f'layer_{i}_bias' in params:
                layer.bias = params[f'layer_{i}_bias']
            if hasattr(layer, 'gamma') and f'layer_{i}_gamma' in params:
                layer.gamma = params[f'layer_{i}_gamma']
            if hasattr(layer, 'beta') and f'layer_{i}_beta' in params:
                layer.beta = params[f'layer_{i}_beta']
            if hasattr(layer, 'running_mean') and f'layer_{i}_running_mean' in params:
                layer.running_mean = params[f'layer_{i}_running_mean']
            if hasattr(layer, 'running_var') and f'layer_{i}_running_var' in params:
                layer.running_var = params[f'layer_{i}_running_var']
        
        print(f"✓ Модель загружена из {path}")

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def softmax(x):
    """Softmax функция"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, labels):
    """Функция потерь - кросс-энтропия"""
    probs = softmax(logits)
    batch_size = logits.shape[0]
    
    # One-hot encoding для меток
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), labels] = 1
    
    loss = -np.sum(one_hot * np.log(probs + 1e-8)) / batch_size
    return loss

def cross_entropy_gradient(logits, labels):
    """Градиент функции потерь"""
    probs = softmax(logits)
    batch_size = logits.shape[0]
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), labels] = 1
    
    grad = (probs - one_hot) / batch_size
    return grad

def preprocess_image(image_path, img_size=224):
    """Предобработка изображения"""
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    
    # Изменение размера
    image = image.resize((img_size, img_size))
    
    # Преобразование в numpy массив и нормализация в [0, 1]
    image = np.array(image) / 255.0
    
    # Стандартизация
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Транспонирование для формата (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    # Добавление размерности батча
    image = np.expand_dims(image, axis=0)
    
    return image

# ==================== КЛАСС ДЛЯ РАСПОЗНАВАНИЯ ====================

class NumPyImageRecognizer:
    """Класс для распознавания изображений с помощью NumPy модели"""
    
    def __init__(self, model_path: str, class_names: List[str]):
        """
        Инициализация распознавателя
        
        Args:
            model_path: путь к файлу с сохраненной моделью
            class_names: список названий классов
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Создание и загрузка модели
        self.model = NumPyImageClassifier(num_classes=self.num_classes)
        self.model.load(model_path)
        
        print(f"✓ Модель загружена. Доступные классы: {class_names}")
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Распознать объект на изображении
        
        Args:
            image_path: путь к изображению
            top_k: количество лучших предсказаний
        
        Returns:
            список словарей с классом и вероятностью
        """
        try:
            # Предобработка изображения
            image = preprocess_image(image_path)
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None
        
        # Прямой проход через модель
        logits = self.model.forward(image, training=False)
        
        # Softmax для получения вероятностей
        probs = softmax(logits)[0]
        
        # Получение top-k индексов
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        # Формирование результатов
        results = []
        for idx in top_indices:
            results.append({
                'class': self.class_names[idx],
                'probability': float(probs[idx]),
                'confidence': f"{probs[idx]*100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Распознать несколько изображений
        
        Args:
            image_paths: список путей к изображениям
        """
        results = []
        for path in image_paths:
            pred = self.predict(path, top_k=1)
            if pred:
                results.append({
                    'image': os.path.basename(path),
                    'prediction': pred[0]['class'],
                    'confidence': pred[0]['confidence']
                })
        return results
    
    def show_prediction(self, image_path: str):
        """Показать изображение с предсказанием"""
        # Получаем предсказание
        results = self.predict(image_path, top_k=1)
        if results is None:
            return
        
        # Загрузка оригинального изображения
        image = Image.open(image_path)
        
        # Получаем вероятности всех классов
        image_tensor = preprocess_image(image_path)
        logits = self.model.forward(image_tensor, training=False)
        probs = softmax(logits)[0]
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Изображение
        ax1.imshow(image)
        ax1.set_title(f"Предсказание: {results[0]['class']}\nУверенность: {results[0]['confidence']}")
        ax1.axis('off')
        
        # Топ-10 вероятностей
        top_k = min(10, len(self.class_names))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_classes = [self.class_names[i] for i in top_indices]
        top_probs = [probs[i] for i in top_indices]
        
        colors = ['green'] + ['gray'] * (top_k - 1)
        ax2.barh(top_classes, top_probs, color=colors)
        ax2.set_xlabel('Вероятность')
        ax2.set_title(f'Топ-{top_k} предсказаний')
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()

# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    # Пример использования NumPy нейросети
    
    MODEL_PATH = "numpy_best_model.pkl"  # Путь к файлу модели
    CLASS_NAMES = ["кошка", "собака", "птица", "рыба", "цветок", 
                   "дерево", "машина", "дом", "книга", "человек"]
    
    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        print(f"Модель {MODEL_PATH} не найдена. Сначала обучите модель с помощью numpy_train.py")
        exit()
    
    # Создаем распознаватель
    recognizer = NumPyImageRecognizer(MODEL_PATH, CLASS_NAMES)
    
    # Пример распознавания
    image_path = "test_image.jpg"
    
    if os.path.exists(image_path):
        print("\n=== Распознавание изображения ===")
        results = recognizer.predict(image_path, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['class']}: {result['confidence']}")
        
        # Показать визуализацию
        recognizer.show_prediction(image_path)
    
    # Интерактивный режим
    print("\n=== Интерактивный режим ===")
    print("Введите путь к изображению (или 'exit' для выхода):")
    
    while True:
        user_input = input("\nПуть к изображению: ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if os.path.exists(user_input):
            results = recognizer.predict(user_input, top_k=3)
            if results:
                print("\nРезультаты:")
                for result in results:
                    print(f"  {result['class']}: {result['confidence']}")
        else:
            print("Файл не найден. Попробуйте снова.")
