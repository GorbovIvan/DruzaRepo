import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from datetime import datetime

# ==================== АРХИТЕКТУРА НЕЙРОСЕТИ ====================
class ImageClassifier(nn.Module):
    """Сверточная нейросеть для классификации изображений"""
    
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        
        # Сверточные слои
        self.features = nn.Sequential(
            # Блок 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== КЛАСС ДЛЯ РАБОТЫ С ДАННЫМИ ====================
class ImageDataset(Dataset):
    """Датасет для загрузки изображений"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== КЛАСС ДЛЯ ОБУЧЕНИЯ ====================
class Trainer:
    """Класс для обучения нейросети"""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Обучение на одной эпохе"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Обучение')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Валидация'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=30, save_path='best_model.pth'):
        """
        Полный цикл обучения
        
        Args:
            train_loader: загрузчик обучающих данных
            val_loader: загрузчик валидационных данных
            epochs: количество эпох
            save_path: путь для сохранения лучшей модели
        """
        best_val_acc = 0
        patience_counter = 0
        early_stop_patience = 7
        
        print(f"Начало обучения на {epochs} эпох")
        print("=" * 60)
        
        for epoch in range(epochs):
            print(f'\nЭпоха {epoch+1}/{epochs}')
            print("-" * 40)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Валидация
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Обновление learning rate
            self.scheduler.step(val_loss)
            
            # Вывод результатов
            print(f"\nРезультаты эпохи {epoch+1}:")
            print(f"  Обучение - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Валидация - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Новая лучшая модель! Точность: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Ранняя остановка
            if patience_counter >= early_stop_patience:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break
        
        print("\n" + "=" * 60)
        print(f"Обучение завершено! Лучшая точность: {best_val_acc:.2f}%")
        
        # Сохранение истории обучения
        self.save_history()
        
        return self.history
    
    def save_history(self, filename='training_history.json'):
        """Сохранение истории обучения"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)
        print(f"История обучения сохранена в {filename}")
    
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
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

# ==================== ФУНКЦИИ ДЛЯ ПОДГОТОВКИ ДАННЫХ ====================

def load_dataset(data_dir):
    """
    Загрузка датасета из папок
    
    Args:
        data_dir: путь к папке с датасетом
    
    Returns:
        image_paths: список путей к изображениям
        labels: список меток
        class_names: список названий классов
    """
    image_paths = []
    labels = []
    class_names = []
    
    # Получаем все классы (папки)
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            
            # Загружаем изображения из папки класса
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(len(class_names) - 1)
    
    return image_paths, labels, class_names

def get_transforms():
    """Получение трансформаций для обучения и валидации"""
    
    # Трансформации для обучения (с аугментацией)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации (без аугментации)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

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
    save_dir = f"training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Проверка доступности GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загрузка данных
    print("\nЗагрузка датасета...")
    if not os.path.exists(DATA_DIR):
        print(f"Ошибка: Папка {DATA_DIR} не найдена!")
        print("Создайте папку 'dataset' со структурой:")
        print("  dataset/")
        print("    класс1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    класс2/")
        print("      image1.jpg")
        print("      image2.jpg")
        return
    
    image_paths, labels, class_names = load_dataset(DATA_DIR)
    
    if len(image_paths) == 0:
        print("Ошибка: Не найдено изображений в датасете!")
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
    
    # Получение трансформаций
    train_transform, val_transform = get_transforms()
    
    # Создание датасетов
    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)
    
    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Создание модели
    print("\nСоздание модели...")
    model = ImageClassifier(num_classes=len(class_names))
    
    # Подсчет количества параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Создание тренера
    trainer = Trainer(model, device, learning_rate=LEARNING_RATE)
    
    # Обучение
    print("\nНачинаем обучение...")
    save_path = os.path.join(save_dir, 'best_model.pth')
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS, save_path=save_path)
    
    # Визуализация результатов
    print("\nВизуализация результатов обучения...")
    trainer.plot_history()
    
    # Сохранение истории в папку
    import shutil
    shutil.move('training_history.json', os.path.join(save_dir, 'training_history.json'))
    shutil.move('training_history.png', os.path.join(save_dir, 'training_history.png'))
    
    print(f"\n✓ Обучение завершено!")
    print(f"✓ Модель сохранена в: {save_path}")
    print(f"✓ Результаты сохранены в папке: {save_dir}")
    
    # Создание файла с информацией об обучении
    training_info = {
        'timestamp': timestamp,
        'device': str(device),
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'validation_split': VAL_SPLIT,
        'num_classes': len(class_names),
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'best_val_acc': max(history['val_acc']),
        'best_epoch': history['val_acc'].index(max(history['val_acc'])) + 1
    }
    
    with open(os.path.join(save_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=4, ensure_ascii=False)

# ==================== ТЕСТИРОВАНИЕ МОДЕЛИ ПОСЛЕ ОБУЧЕНИЯ ====================

def test_model(model_path, class_names, test_image_path):
    """Тестирование обученной модели на одном изображении"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка модели
    model = ImageClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка и предсказание
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, 3)
    
    # Вывод результатов
    print("\nРезультаты тестирования:")
    for i in range(3):
        class_idx = top_class[0][i].item()
        prob = top_prob[0][i].item()
        print(f"  {i+1}. {class_names[class_idx]}: {prob*100:.2f}%")
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.imshow(image)
    ax1.set_title(f"Предсказание: {class_names[top_class[0][0].item()]}")
    ax1.axis('off')
    
    probs = probabilities[0].cpu().numpy()
    ax2.barh(class_names, probs)
    ax2.set_xlabel('Вероятность')
    ax2.set_title('Вероятности классов')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Запуск обучения
    main()
    
    # Раскомментируйте для тестирования после обучения
    # test_model('best_model.pth', class_names, 'test_image.jpg')
