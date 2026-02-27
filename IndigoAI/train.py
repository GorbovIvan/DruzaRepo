import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import json
from datetime import datetime

# Импортируем классы из первого файла
from numpy_inference import (
    NumPyImageClassifier, Conv2D, Dense, MaxPool2D, ReLU, Flatten,
    BatchNorm2D, Dropout, GlobalAveragePooling2D,
    softmax, cross_entropy_loss, cross_entropy_gradient, preprocess_image
)

# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ ====================

def load_dataset(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Загрузка датасета из папок
    """
    image_paths = []
    labels = []
    class_names = []
    
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(len(class_names) - 1)
    
    return image_paths, labels, class_names

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Аугментация изображения для обучения
    
    Args:
        image: изображение в формате (C, H, W)
    
    Returns:
        аугментированное изображение
    """
    # Переводим в формат (H, W, C) для операций
    image = np.transpose(image, (1, 2, 0))
    
    # Случайное горизонтальное отражение
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Случайное вращение (упрощенное - только кратные 90 градусам)
    if np.random.random() > 0.7:
        k = np.random.randint(1, 4)
        image = np.rot90(image, k)
    
    # Случайное изменение яркости
    if np.random.random() > 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
    
    # Добавление шума
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 0.01, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    # Обратно в формат (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    return image

def load_and_preprocess_image(path: str, augment: bool = False) -> np.ndarray:
    """
    Загрузка и предобработка изображения
    
    Args:
        path: путь к изображению
        augment: применять ли аугментацию
    
    Returns:
        тензор изображения в формате (C, H, W)
    """
    # Загрузка
    img = Image.open(path).convert('RGB')
    
    # Изменение размера
    img = img.resize((224, 224))
    
    # В массив и нормализация
    img = np.array(img) / 255.0
    
    # Аугментация
    if augment:
        img = augment_image(np.transpose(img, (2, 0, 1)))
        img = np.transpose(img, (1, 2, 0))
    
    # Стандартизация
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Транспонирование для формата (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    return img

def create_batches(image_paths: List[str], labels: List[int], 
                   batch_size: int, augment: bool = False):
    """
    Генератор батчей для обучения
    """
    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    
    while True:
        if augment:
            np.random.shuffle(indices)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img = load_and_preprocess_image(image_paths[idx], augment)
                batch_images.append(img)
                batch_labels.append(labels[idx])
            
            yield np.array(batch_images), np.array(batch_labels)

# ==================== КЛАСС ДЛЯ ОБУЧЕНИЯ ====================

class NumPyTrainer:
    """Класс для обучения модели на NumPy"""
    
    def __init__(self, model: NumPyImageClassifier, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_gen, num_batches: int) -> Tuple[float, float]:
        """
        Обучение на одной эпохе
        """
        epoch_loss = []
        epoch_acc = []
        
        for _ in range(num_batches):
            # Получение батча
            batch_images, batch_labels = next(train_gen)
            
            # Прямой проход
            logits = self.model.forward(batch_images, training=True)
            
            # Вычисление потерь
            loss = cross_entropy_loss(logits, batch_labels)
            
            # Вычисление точности
            predictions = np.argmax(logits, axis=1)
            acc = np.mean(predictions == batch_labels)
            
            # Обратный проход
            grad = cross_entropy_gradient(logits, batch_labels)
            self.model.backward(grad, self.learning_rate)
            
            epoch_loss.append(loss)
            epoch_acc.append(acc)
        
        return np.mean(epoch_loss), np.mean(epoch_acc)
    
    def validate(self, val_gen, num_batches: int) -> Tuple[float, float]:
        """
        Валидация модели
        """
        val_loss = []
        val_acc = []
        
        for _ in range(num_batches):
            batch_images, batch_labels = next(val_gen)
            
            logits = self.model.forward(batch_images, training=False)
            loss = cross_entropy_loss(logits, batch_labels)
            predictions = np.argmax(logits, axis=1)
            acc = np.mean(predictions == batch_labels)
            
            val_loss.append(loss)
            val_acc.append(acc)
        
        return np.mean(val_loss), np.mean(val_acc)
    
    def train(self, train_paths: List[str], train_labels: List[int],
              val_paths: List[str], val_labels: List[int],
              batch_size: int = 32, epochs: int = 30, 
              save_path: str = 'numpy_best_model.pkl'):
        """
        Полный цикл обучения
        """
        # Создание генераторов
        train_gen = create_batches(train_paths, train_labels, batch_size, augment=True)
        val_gen = create_batches(val_paths, val_labels, batch_size, augment=False)
        
        num_train_batches = len(train_paths) // batch_size
        num_val_batches = len(val_paths) // batch_size
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 7
        
        print(f"Начало обучения на {epochs} эпох")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Обучение
            train_loss, train_acc = self.train_epoch(train_gen, num_train_batches)
            
            # Валидация
            val_loss, val_acc = self.validate(val_gen, num_val_batches)
            
            # Сохранение истории
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc * 100))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc * 100))
            
            # Вывод результатов
            print(f"\nЭпоха {epoch+1}/{epochs}")
            print(f"  Обучение - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"  Валидация - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save(save_path)
                print(f"  ✓ Новая лучшая модель! Точность: {val_acc*100:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Ранняя остановка
            if patience_counter >= early_stop_patience:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break
            
            # Уменьшение learning rate при плато
            if patience_counter >= 3:
                self.learning_rate *= 0.5
                print(f"  Learning rate уменьшен до {self.learning_rate:.6f}")
        
        print("\n" + "=" * 60)
        print(f"Обучение завершено! Лучшая точность: {best_val_acc*100:.2f}%")
        
        return self.history
    
    def plot_history(self):
        """Визуализация процесса обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(self.history['train_loss'], label='Обучение', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Валидация', linewidth=2)
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Потери (Loss)')
        ax1.set_title('Динамика потерь')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График точности
        ax2.plot(self.history['train_acc'], label='Обучение', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Валидация', linewidth=2)
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Точность (%)')
        ax2.set_title('Динамика точности')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('numpy_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

# ==================== ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ====================

def main():
    """Основная функция для запуска обучения"""
    
    # Параметры обучения
    DATA_DIR = "dataset"  # Путь к папке с датасетом
    BATCH_SIZE = 16  # Уменьшаем для NumPy (из-за памяти)
    EPOCHS = 30
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Создание папки для сохранения результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"numpy_training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Установка случайного seed
    np.random.seed(RANDOM_SEED)
    
    # Загрузка данных
    print("\nЗагрузка датасета...")
    if not os.path.exists(DATA_DIR):
        print(f"Ошибка: Папка {DATA_DIR} не найдена!")
        print("Создайте папку 'dataset' со структурой:")
        print("  dataset/")
        print("    класс1/")
        print("      image1.jpg")
        print("    класс2/")
        return
    
    image_paths, labels, class_names = load_dataset(DATA_DIR)
    
    if len(image_paths) == 0:
        print("Ошибка: Не найдено изображений!")
        return
    
    print(f"Найдено изображений: {len(image_paths)}")
    print(f"Количество классов: {len(class_names)}")
    print(f"Классы: {class_names}")
    
    # Сохранение информации о классах
    class_info = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'num_images': len(image_paths)
    }
    with open(os.path.join(save_dir, 'class_info.json'), 'w', encoding='utf-8') as f:
        json.dump(class_info, f, indent=4, ensure_ascii=False)
    
    # Разделение на обучающую и валидационную выборки
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=VAL_SPLIT, 
        random_state=RANDOM_SEED,
        stratify=labels
    )
    
    print(f"\nРазмер обучающей выборки: {len(train_paths)}")
    print(f"Размер валидационной выборки: {len(val_paths)}")
    
    # Создание модели
    print("\nСоздание модели...")
    model = NumPyImageClassifier(num_classes=len(class_names))
    
    # Подсчет количества параметров
    total_params = sum(layer.weights.size for layer in model.layers 
                      if hasattr(layer, 'weights'))
    print(f"Всего параметров: {total_params:,}")
    
    # Создание тренера
    trainer = NumPyTrainer(model, learning_rate=LEARNING_RATE)
    
    # Обучение
    print("\nНачинаем обучение...")
    save_path = os.path.join(save_dir, 'numpy_best_model.pkl')
    history = trainer.train(
        train_paths, train_labels,
        val_paths, val_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        save_path=save_path
    )
    
    # Визуализация результатов
    print("\nВизуализация результатов обучения...")
    trainer.plot_history()
    
    # Сохранение истории
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    
    # Сохранение информации об обучении
    training_info = {
        'timestamp': timestamp,
        'framework': 'NumPy (pure)',
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'validation_split': VAL_SPLIT,
        'num_classes': len(class_names),
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'best_val_acc': float(max(history['val_acc'])),
        'best_epoch': int(np.argmax(history['val_acc']) + 1)
    }
    
    with open(os.path.join(save_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ Обучение завершено!")
    print(f"✓ Модель сохранена в: {save_path}")
    print(f"✓ Результаты сохранены в папке: {save_dir}")
    
    # Сохраняем также классы отдельно для удобства
    with open(os.path.join(save_dir, 'class_names.json'), 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

# ==================== ТЕСТИРОВАНИЕ МОДЕЛИ ====================

def test_model(model_path: str, class_names: List[str], test_image_path: str):
    """Тестирование обученной модели"""
    
    # Загрузка модели
    model = NumPyImageClassifier(num_classes=len(class_names))
    model.load(model_path)
    
    # Предобработка изображения
    image = preprocess_image(test_image_path)
    
    # Предсказание
    logits = model.forward(image, training=False)
    probs = softmax(logits)[0]
    
    # Топ-3 предсказания
    top_indices = np.argsort(probs)[::-1][:3]
    
    print("\nРезультаты тестирования:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {class_names[idx]}: {probs[idx]*100:.2f}%")
    
    # Визуализация
    image = Image.open(test_image_path)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Тестовое изображение")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    probs_list = probs.tolist()
    classes = class_names
    plt.barh(classes, probs_list)
    plt.xlabel('Вероятность')
    plt.title('Предсказания модели')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Запуск обучения
    main()
    
    # Раскомментируйте для тестирования после обучения
    # test_model('numpy_best_model.pkl', class_names, 'test_image.jpg')
