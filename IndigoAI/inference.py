import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== АРХИТЕКТУРА НЕЙРОСЕТИ ====================
class ImageClassifier(nn.Module):
    """Сверточная нейросеть для классификации изображений"""
    
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        
        # Сверточные слои для извлечения признаков
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

# ==================== КЛАСС ДЛЯ РАСПОЗНАВАНИЯ ====================
class ImageRecognizer:
    """Класс для распознавания изображений с помощью обученной модели"""
    
    def __init__(self, model_path, class_names):
        """
        Инициализация распознавателя
        
        Args:
            model_path: путь к файлу с обученной моделью
            class_names: список названий классов
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        
        # Загрузка модели
        print(f"Загрузка модели с {self.device}...")
        self.model = ImageClassifier(num_classes=len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Модель успешно загружена!")
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=3):
        """
        Распознать объект на изображении
        
        Args:
            image_path: путь к изображению
            top_k: количество лучших предсказаний
        
        Returns:
            список словарей с классом и вероятностью
        """
        # Загрузка и подготовка изображения
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None
        
        # Применение трансформаций
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Формирование результатов
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            probability = top_probs[0][i].item()
            results.append({
                'class': self.class_names[class_idx],
                'probability': probability,
                'confidence': f"{probability*100:.2f}%"
            })
        
        return results
    
    def predict_from_bytes(self, image_bytes, top_k=3):
        """
        Распознать объект из байтов изображения
        
        Args:
            image_bytes: байты изображения
            top_k: количество лучших предсказаний
        """
        from io import BytesIO
        
        # Загрузка изображения из байтов
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Формирование результатов
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            probability = top_probs[0][i].item()
            results.append({
                'class': self.class_names[class_idx],
                'probability': probability,
                'confidence': f"{probability*100:.2f}%"
            })
        
        return results
    
    def show_prediction(self, image_path):
        """
        Показать изображение с предсказанием
        
        Args:
            image_path: путь к изображению
        """
        # Получаем предсказание
        results = self.predict(image_path, top_k=1)
        if results is None:
            return
        
        # Загрузка изображения
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
            classes = [p['class'] for p in all_probs]
            probs = [p['probability'] for p in all_probs]
            
            colors = ['green' if i == 0 else 'gray' for i in range(len(classes))]
            ax2.barh(classes, probs, color=colors)
            ax2.set_xlabel('Вероятность')
            ax2.set_title('Вероятности всех классов')
            ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def predict_all_probabilities(self, image_path):
        """
        Получить вероятности для всех классов
        
        Args:
            image_path: путь к изображению
        """
        # Загрузка и подготовка изображения
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Сортировка по убыванию вероятности
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'class': self.class_names[i],
                'probability': prob.item()
            })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)
    
    def predict_batch(self, image_paths):
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

# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================
if __name__ == "__main__":
    # Пример использования нейросети
    
    # Укажите путь к обученной модели и список классов
    MODEL_PATH = "best_model.pth"  # Путь к файлу модели
    CLASS_NAMES = ["кошка", "собака", "птица", "рыба", "цветок", 
                   "дерево", "машина", "дом", "книга", "человек"]  # Замените на свои классы
    
    # Создаем распознаватель
    recognizer = ImageRecognizer(MODEL_PATH, CLASS_NAMES)
    
    # Пример 1: Распознать одно изображение
    image_path = "test_image.jpg"  # Замените на путь к вашему изображению
    
    if os.path.exists(image_path):
        print("\n=== Распознавание изображения ===")
        results = recognizer.predict(image_path, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['class']}: {result['confidence']}")
        
        # Показать визуализацию
        recognizer.show_prediction(image_path)
    
    # Пример 2: Пакетное распознавание
    image_folder = "test_images"  # Папка с тестовыми изображениями
    if os.path.exists(image_folder):
        test_images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if test_images:
            print("\n=== Пакетное распознавание ===")
            batch_results = recognizer.predict_batch(test_images)
            for result in batch_results:
                print(f"{result['image']}: {result['prediction']} ({result['confidence']})")
    
    # Пример 3: Интерактивный режим
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
