import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
import pickle

# ==================== АРХИТЕКТУРА НЕЙРОСЕТИ НА FLAX ====================
class ImageClassifier(nn.Module):
    """Сверточная нейросеть на Flax для классификации изображений"""
    num_classes: int
    
    @nn.compact
    def __call__(self, x, training=True):
        # Сверточные слои
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Глобальный пулинг и классификатор
        x = jnp.mean(x, axis=(1, 2))  # Global Average Pooling
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.3, deterministic=not training)(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def normalize_image(image):
    """Нормализация изображения"""
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
    return (image - mean) / std

def preprocess_image(image_path, img_size=224):
    """
    Предобработка изображения для подачи в модель
    
    Args:
        image_path: путь к изображению
        img_size: размер выходного изображения
    
    Returns:
        нормализованный тензор изображения
    """
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    
    # Изменение размера
    image = image.resize((img_size, img_size))
    
    # Преобразование в numpy массив и нормализация в [0, 1]
    image = np.array(image) / 255.0
    
    # Транспонирование для формата (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    # Добавление размерности батча
    image = np.expand_dims(image, axis=0)
    
    # Преобразование в JAX массив и нормализация
    image = jnp.array(image)
    image = normalize_image(image)
    
    return image

# ==================== КЛАСС ДЛЯ РАСПОЗНАВАНИЯ ====================
class JAXImageRecognizer:
    """Класс для распознавания изображений с помощью JAX модели"""
    
    def __init__(self, model_path: str, class_names: List[str]):
        """
        Инициализация распознавателя
        
        Args:
            model_path: путь к файлу с сохраненной моделью
            class_names: список названий классов
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Создание модели
        self.model = ImageClassifier(num_classes=self.num_classes)
        
        # Инициализация параметров (нужна только для получения структуры)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 3, 224, 224))
        self.params = self.model.init(rng, dummy_input, training=False)['params']
        
        # Загрузка сохраненных параметров
        self.load_model(model_path)
        
        # Компиляция функции предсказания
        self.predict_fn = jax.jit(self._predict)
        print(f"✓ Модель загружена. Доступные классы: {class_names}")
    
    def load_model(self, model_path: str):
        """Загрузка параметров модели"""
        with open(model_path, 'rb') as f:
            saved_params = pickle.load(f)
        self.params = saved_params
        print(f"✓ Модель загружена из {model_path}")
    
    def _predict(self, params, image):
        """Функция предсказания (будет скомпилирована JIT)"""
        logits = self.model.apply({'params': params}, image, training=False)
        probabilities = jax.nn.softmax(logits)
        return probabilities
    
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
        
        # Предсказание
        probabilities = self.predict_fn(self.params, image)
        probabilities = probabilities[0]  # Убираем размерность батча
        
        # Получение top-k индексов
        top_indices = jnp.argsort(probabilities)[::-1][:top_k]
        
        # Формирование результатов
        results = []
        for idx in top_indices:
            idx = int(idx)
            prob = float(probabilities[idx])
            results.append({
                'class': self.class_names[idx],
                'probability': prob,
                'confidence': f"{prob*100:.2f}%"
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
    
    def predict_all_probabilities(self, image_path: str) -> List[Dict[str, Any]]:
        """Получить вероятности для всех классов"""
        try:
            image = preprocess_image(image_path)
        except:
            return None
        
        probabilities = self.predict_fn(self.params, image)[0]
        
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'class': self.class_names[i],
                'probability': float(prob)
            })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)
    
    def show_prediction(self, image_path: str):
        """Показать изображение с предсказанием"""
        # Получаем предсказание
        results = self.predict(image_path, top_k=1)
        if results is None:
            return
        
        # Загрузка оригинального изображения
        image = Image.open(image_path)
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Изображение
        ax1.imshow(image)
        ax1.set_title(f"Предсказание: {results[0]['class']}\nУверенность: {results[0]['confidence']}")
        ax1.axis('off')
        
        # Вероятности всех классов
        all_probs = self.predict_all_probabilities(image_path)
        if all_probs:
            classes = [p['class'] for p in all_probs[:10]]  # Топ-10
            probs = [p['probability'] for p in all_probs[:10]]
            
            colors = ['green' if i == 0 else 'gray' for i in range(len(classes))]
            ax2.barh(classes, probs, color=colors)
            ax2.set_xlabel('Вероятность')
            ax2.set_title('Топ-10 вероятностей')
            ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()

# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================
if __name__ == "__main__":
    # Пример использования JAX нейросети
    
    MODEL_PATH = "jax_best_model.pkl"  # Путь к файлу модели
    CLASS_NAMES = ["кошка", "собака", "птица", "рыба", "цветок", 
                   "дерево", "машина", "дом", "книга", "человек"]
    
    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        print(f"Модель {MODEL_PATH} не найдена. Сначала обучите модель с помощью jax_train.py")
        exit()
    
    # Создаем распознаватель
    recognizer = JAXImageRecognizer(MODEL_PATH, CLASS_NAMES)
    
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
