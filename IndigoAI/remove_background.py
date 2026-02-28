import os
import sys
from pathlib import Path
from rembg import remove
from PIL import Image

def remove_background_from_folder(input_folder, output_folder=None):
    """
    Удаляет фон из всех изображений в указанной папке
    
    Args:
        input_folder (str): Путь к папке с исходными изображениями
        output_folder (str): Путь для сохранения обработанных изображений
    """
    
    # Создаем папку для результатов, если она не указана
    if output_folder is None:
        output_folder = os.path.join(input_folder, "no_bg")
    
    # Создаем выходную папку, если её нет
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Поддерживаемые форматы изображений
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Получаем список всех файлов в папке
    files = os.listdir(input_folder)
    image_files = [f for f in files if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print("В папке не найдено изображений поддерживаемых форматов")
        return
    
    print(f"Найдено изображений: {len(image_files)}")
    
    # Обрабатываем каждое изображение
    for i, filename in enumerate(image_files, 1):
        try:
            input_path = os.path.join(input_folder, filename)
            
            # Формируем имя выходного файла
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_no_bg.png"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"[{i}/{len(image_files)}] Обработка: {filename}")
            
            # Открываем и обрабатываем изображение
            with open(input_path, 'rb') as inp_file:
                input_data = inp_file.read()
                output_data = remove(input_data)
            
            # Сохраняем результат
            with open(output_path, 'wb') as out_file:
                out_file.write(output_data)
            
            print(f"  ✓ Сохранено как: {output_filename}")
            
        except Exception as e:
            print(f"  ✗ Ошибка при обработке {filename}: {str(e)}")
    
    print(f"\nГотово! Обработанные изображения сохранены в: {output_folder}")

if __name__ == "__main__":
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Если аргументы не переданы, запрашиваем путь у пользователя
        input_folder = input("Введите путь к папке с изображениями: ").strip()
        output_folder = input("Введите путь для сохранения (Enter для папки 'no_bg'): ").strip() or None
    
    # Удаляем кавычки, если они есть
    input_folder = input_folder.strip('"\'')
    
    if not os.path.exists(input_folder):
        print("Указанная папка не существует!")
        sys.exit(1)
    
    remove_background_from_folder(input_folder, output_folder)
