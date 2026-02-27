import os
import glob

def rename_jpeg_to_dog(folder_path):
    """
    Переименовывает все JPEG файлы в указанной папке в формат dog_1.jpeg, dog_2.jpeg и т.д.
    
    Args:
        folder_path (str): Путь к папке с фотографиями
    """
    
    # Поиск всех JPEG файлов (регистронезависимый)
    jpeg_files = []
    
    # Различные варианты расширений JPEG
    extensions = ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG']
    
    for ext in extensions:
        jpeg_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Сортировка файлов для последовательной нумерации
    jpeg_files.sort()
    
    if not jpeg_files:
        print("JPEG файлы не найдены в указанной папке")
        return
    
    print(f"Найдено {len(jpeg_files)} JPEG файлов")
    
    # Переименование файлов
    for index, file_path in enumerate(jpeg_files, start=1):
        # Получаем директорию и расширение файла
        directory = os.path.dirname(file_path)
        file_extension = os.path.splitext(file_path)[1]
        
        # Новое имя файла
        new_name = f"dog_{index}{file_extension}"
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(file_path, new_path)
            print(f"Переименован: {os.path.basename(file_path)} -> {new_name}")
        except Exception as e:
            print(f"Ошибка при переименовании {file_path}: {e}")

def main():
    # Путь к папке с фотографиями
    # Можно изменить на нужный путь или запросить у пользователя
    folder_path = input("Введите путь к папке с фотографиями: ").strip()
    
    # Убираем кавычки, если пользователь их вставил
    folder_path = folder_path.strip('"\'')
    
    # Проверяем существование папки
    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не существует")
        return
    
    if not os.path.isdir(folder_path):
        print(f"Ошибка: '{folder_path}' не является папкой")
        return
    
    # Запрос подтверждения
    print(f"\nБудут переименованы все JPEG файлы в папке: {folder_path}")
    confirm = input("Продолжить? (y/n): ").strip().lower()
    
    if confirm == 'y':
        rename_jpeg_to_dog(folder_path)
        print("Готово!")
    else:
        print("Операция отменена")

if __name__ == "__main__":
    main()
