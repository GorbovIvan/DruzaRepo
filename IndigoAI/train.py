import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import json
from datetime import datetime

# ==================== АРХИТЕКТУРА НЕЙРОСЕТИ ====================
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

def preprocess_image(image_path: str, img_size: int = 224, augment: bool = False) -> jnp.ndarray:
    """
    Предобработка изображения с опциональной аугментацией
    """
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    
    # Аугментация для обучения
    if augment:
        # Случайное горизонтальное отражение
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Случайное вращение
        angle = np.random.randint(-10, 10)
        image = image.rotate(angle, expand=False)
        
        # Случайное изменение яркости
        brightness = np.random.uniform(0.8, 1.2)
        image = image.point(lambda p: p * brightness)
    
    # Изменение размера
    image = image.resize((img_size, img_size))
    
    # Преобразование в массив и нормализация
    image = np.array(image) / 255.0
    
    # Стандартизация
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Транспонирование для формата (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    return jnp.array(image)

def create_batches(image_paths: List[str], labels: List[int], 
                   batch_size: int, augment: bool = False):
    """
    Создание батчей для обучения
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
                img = preprocess_image(image_paths[idx], augment=augment)
                batch_images.append(img)
                batch_labels.append(labels[idx])
            
            yield jnp.stack(batch_images), jnp.array(batch_labels)

# ==================== ФУНКЦИИ ПОТЕРЬ И МЕТРИК ====================

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Функция потерь - кросс-энтропия"""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Вычисление точности"""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# ==================== ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ ====================

def create_train_state(rng, model, learning_rate: float, num_classes: int):
    """Создание состояния обучения"""
    dummy_input = jnp.ones((1, 3, 224, 224))
    params = model.init(rng, dummy_input, training=False)['params']
    
    # Создание оптимизатора
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)
    
    return params, opt_state, tx

@jax.jit
def train_step(params, opt_state, tx, batch_images, batch_labels):
    """Один шаг обучения (JIT-компилируется)"""
    
    def loss_fn(params):
        logits = model.apply({'params': params}, batch_images, training=True)
        loss = cross_entropy_loss(logits, batch_labels)
        return loss, logits
    
    # Вычисление градиентов
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Обновление параметров
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Вычисление метрик
    accuracy = compute_accuracy(logits, batch_labels)
    
    return params, opt_state, loss, accuracy

@jax.jit
def eval_step(params, batch_images, batch_labels):
    """Шаг валидации"""
    logits = model.apply({'params': params}, batch_images, training=False)
    loss = cross_entropy_loss(logits, batch_labels)
    accuracy = compute_accuracy(logits, batch_labels)
    return loss, accuracy

# ==================== КЛАСС ДЛЯ ОБУЧЕНИЯ ====================
class JAXTrainer:
    """Класс для обучения модели на JAX"""
    
    def __init__(self, model, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train(self, train_paths: List[str], train_labels: List[int],
              val_paths: List[str], val_labels: List[int],
              num_classes: int, batch_size: int = 32,
              epochs: int = 30, save_path: str = 'jax_best_model.pkl'):
        """
        Обучение модели
        """
        # Инициализация
        rng = jax.random.PRNGKey(0)
        self.params, self.opt_state, self.tx = create_train_state(
            rng, self.model, self.learning_rate, num_classes
        )
        
        # Создание генераторов батчей
        train_gen = create_batches(train_paths, train_labels, batch_size, augment=True)
        val_gen = create_batches(val_paths, val_labels, batch_size, augment=False)
        
        # Метрики для валидации
        num_val_batches = len(val_paths) // batch_size + 1
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 7
        
        print(f"Начало обучения на {epochs} эпох")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Обучение
            epoch_train_loss = []
            epoch_train_acc = []
            
            num_batches = len(train_paths) // batch_size
            pbar = tqdm(range(num_batches), desc=f'Эпоха {epoch+1}/{epochs}')
            
            for _ in pbar:
                batch_images, batch_labels = next(train_gen)
                
                self.params, self.opt_state, loss, acc = train_step(
                    self.params, self.opt_state, self.tx, 
                    batch_images, batch_labels
                )
                
                epoch_train_loss.append(float(loss))
                epoch_train_acc.append(float(acc))
                
                pbar.set_postfix({
                    'loss': f'{np.mean(epoch_train_loss[-10:]):.4f}',
                    'acc': f'{np.mean(epoch_train_acc[-10:]):.2f}%'
                })
            
            avg_train_loss = np.mean(epoch_train_loss)
            avg_train_acc = np.mean(epoch_train_acc)
            
            # Валидация
            val_losses = []
            val_accs = []
            
            for _ in range(num_val_batches):
                try:
                    batch_images, batch_labels = next(val_gen)
                    loss, acc = eval_step(self.params, batch_images, batch_labels)
                    val_losses.append(float(loss))
                    val_accs.append(float(acc))
                except StopIteration:
                    break
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)
            
            # Сохранение истории
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(avg_val_acc)
            
            # Вывод результатов
            print(f"\nРезультаты эпохи {epoch+1}:")
            print(f"  Обучение - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%")
            print(f"  Валидация - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}%")
            
            # Сохранение лучшей модели
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                self.save_model(save_path)
                print(f"  ✓ Новая лучшая модель! Точность: {avg_val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Ранняя остановка
            if patience_counter >= early_stop_patience:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break
        
        print("\n" + "=" * 60)
        print(f"Обучение завершено! Лучшая точность: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, path: str):
        """Сохранение модели"""
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"✓ Модель сохранена в {path}")
    
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
        plt.savefig('jax_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

# ==================== ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ====================

def main():
    """Основная функция для запуска обучения"""
    
    # Параметры обучения
    DATA_DIR = "dataset"  # Путь к папке с датасетом
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Создание папки для сохранения результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"jax_training_{timestamp}"
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
    model = ImageClassifier(num_classes=len(class_names))
    
    # Создание тренера
    trainer = JAXTrainer(model, learning_rate=LEARNING_RATE)
    
    # Обучение
    print("\nНачинаем обучение...")
    save_path = os.path.join(save_dir, 'jax_best_model.pkl')
    history = trainer.train(
        train_paths, train_labels,
        val_paths, val_labels,
        num_classes=len(class_names),
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
        'framework': 'JAX/Flax',
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

# ==================== ТЕСТИРОВАНИЕ МОДЕЛИ ====================

def test_model(model_path: str, class_names: List[str], test_image_path: str):
    """Тестирование обученной модели"""
    
    # Создание модели
    model = ImageClassifier(num_classes=len(class_names))
    
    # Загрузка параметров
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    
    # Функция предсказания
    @jax.jit
    def predict(params, image):
        logits = model.apply({'params': params}, image, training=False)
        return jax.nn.softmax(logits)
    
    # Предобработка изображения
    from jax_inference import preprocess_image
    image = preprocess_image(test_image_path)
    
    # Предсказание
    probabilities = predict(params, image)[0]
    
    # Топ-3 предсказания
    top_indices = jnp.argsort(probabilities)[::-1][:3]
    
    print("\nРезультаты тестирования:")
    for i, idx in enumerate(top_indices):
        idx = int(idx)
        prob = float(probabilities[idx])
        print(f"{i+1}. {class_names[idx]}: {prob*100:.2f}%")
    
    # Визуализация
    image = Image.open(test_image_path)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Тестовое изображение")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    probs = np.array(probabilities)
    classes = class_names
    plt.barh(classes, probs)
    plt.xlabel('Вероятность')
    plt.title('Предсказания модели')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Запуск обучения
    main()
    
    # Раскомментируйте для тестирования после обучения
    # test_model('jax_best_model.pkl', class_names, 'test_image.jpg')
