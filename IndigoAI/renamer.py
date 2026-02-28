import os
import glob

def rename_images_to_dog(folder_path, include_png=False):
    """
    Переименовывает все JPEG и опционально PNG файлы в формат dog_1.jpeg, dog_2.jpeg и т.д.
    
    Args:
        folder_path (str): Путь к папке с фотографиями
        include_png (bool): Включать ли PNG файлы в переименование
    """
    
    # Поиск всех файлов изображений
    image_files = []
    
    # Расширения для поиска
    extensions = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG']
    
    # Добавляем PNG если нужно
    if include_png:
        extensions.extend(['*.png', '*.PNG'])
    
    # Собираем все файлы
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Сортировка файлов для последовательной нумерации
    image_files.sort()
    
    if not image_files:
        print("Файлы изображений не найдены в указанной папке")
        return
    
    # Подсчет файлов по типам
    jpeg_count = sum(1 for f in image_files if f.lower().endswith(('.jpg', '.jpeg')))
    png_count = sum(1 for f in image_files if f.lower().endswith('.png'))
    
    print(f"Найдено файлов: всего {len(image_files)}")
    if jpeg_count > 0:
        print(f"  - JPEG: {jpeg_count}")
    if png_count > 0:
        print(f"  - PNG: {png_count}")
    print()
    
    # Переименование файлов
    for index, file_path in enumerate(image_files, start=1):
        # Получаем директорию и расширение файла
        directory = os.path.dirname(file_path)
        file_extension = os.path.splitext(file_path)[1]
        
        # Новое имя файла (сохраняем оригинальное расширение)
        new_name = f"dog_{index}{file_extension}"
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(file_path, new_path)
            print(f"Переименован: {os.path.basename(file_path)} -> {new_name}")
        except Exception as e:
            print(f"Ошибка при переименовании {file_path}: {e}")

def main():
    print("=== Переименование изображений ===")
    
    # Путь к папке с фотографиями
    folder_path = input("Введите путь к папке с изображениями: ").strip()
    
    # Убираем кавычки, если пользователь их вставил
    folder_path = folder_path.strip('"\'')
    
    # Проверяем существование папки
    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не существует")
        return
    
    if not os.path.isdir(folder_path):
        print(f"Ошибка: '{folder_path}' не является папкой")
        return
    
    # Спрашиваем про PNG
    print("\nДоступные форматы:")
    print("1. Только JPEG (включая .jpg, .jpeg)")
    print("2. JPEG и PNG")
    
    format_choice = input("Выберите формат (1 или 2): ").strip()
    
    include_png = (format_choice == "2")
    
    # Показываем, какие файлы будут обработаны
    print(f"\nБудут переименованы:")
    print(f"  - JPEG файлы: Да")
    print(f"  - PNG файлы: {'Да' if include_png else 'Нет'}")
    print(f"В папке: {folder_path}")
    
    # Запрос подтверждения
    confirm = input("\nПродолжить? (y/n): ").strip().lower()
    
    if confirm == 'y':
        rename_images_to_dog(folder_path, include_png)
        print("\nГотово!")
    else:
        print("Операция отменена")

if __name__ == "__main__":
    main()
