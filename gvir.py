#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import os
import random
import pyautogui
import threading
import subprocess
import platform
import math

# Для работы с курсором может понадобиться:
# pip install pyautogui pillow

class MouseJoker:
    def __init__(self):
        self.joke_active = True
        self.screen_width, self.screen_height = pyautogui.size()
        self.system = platform.system()
        self.color_cycle = 0
        self.colors = ['🔴', '⚫', '🟢']  # Красный, Черный, Зеленый
        
    def simulate_color_change(self):
        """Имитирует смену цвета курсора через сообщения и эффекты"""
        self.color_cycle = (self.color_cycle + 1) % 3
        color_name = ['КРАСНЫЙ', 'ЧЁРНЫЙ', 'ЗЕЛЁНЫЙ'][self.color_cycle]
        color_emoji = self.colors[self.color_cycle]
        
        # Показываем сообщение о смене цвета
        print(f"\r{color_emoji} [СИСТЕМА] Цвет курсора: {color_name} {color_emoji}" + " " * 20)
        
        # Делаем небольшую паузу и дёрганье при смене цвета
        for _ in range(3):
            x, y = pyautogui.position()
            pyautogui.moveTo(x + random.randint(-10, 10), y + random.randint(-10, 10), duration=0.05)
            time.sleep(0.05)
        
        return color_name
    
    def rgb_flash_effect(self):
        """Создаёт эффект мерцания RGB"""
        if random.random() < 0.2:  # 20% шанс
            # Быстрая смена цветов
            for i in range(6):
                color_idx = i % 3
                color = self.colors[color_idx]
                print(f"\r{color} RGB-ЭФФЕКТ {color}" + " " * 20, end='', flush=True)
                
                # Дёргаем курсором под каждый цвет
                x, y = pyautogui.position()
                pyautogui.moveTo(
                    x + random.randint(-15, 15), 
                    y + random.randint(-15, 15), 
                    duration=0.02
                )
                time.sleep(0.1)
    
    def move_mouse_randomly(self):
        """Двигает курсор с учётом текущего цвета"""
        while self.joke_active:
            current_x, current_y = pyautogui.position()
            
            # Разная скорость для разных цветов
            if self.color_cycle == 0:  # Красный - быстро
                speed = 0.1
                range_mult = 50
            elif self.color_cycle == 1:  # Чёрный - медленно
                speed = 0.3
                range_mult = 20
            else:  # Зелёный - средне
                speed = 0.15
                range_mult = 35
            
            new_x = current_x + random.randint(-range_mult, range_mult)
            new_y = current_y + random.randint(-range_mult, range_mult)
            
            new_x = max(0, min(new_x, self.screen_width))
            new_y = max(0, min(new_y, self.screen_height))
            
            pyautogui.moveTo(new_x, new_y, duration=speed)
            time.sleep(random.uniform(0.2, 1.0))
    
    def make_cursor_jumpy(self):
        """Заставляет курсор "дрожать" с учётом цвета"""
        if random.random() < 0.4:
            original_x, original_y = pyautogui.position()
            
            # Цвет влияет на характер дрожания
            if self.color_cycle == 0:  # Красный - хаотичный
                jumps = random.randint(8, 15)
                range_val = 25
            elif self.color_cycle == 1:  # Чёрный - плавный
                jumps = random.randint(3, 6)
                range_val = 10
            else:  # Зелёный - средний
                jumps = random.randint(5, 10)
                range_val = 18
            
            for i in range(jumps):
                offset_x = random.randint(-range_val, range_val)
                offset_y = random.randint(-range_val, range_val)
                pyautogui.moveTo(original_x + offset_x, original_y + offset_y, duration=0.03)
                
                # Мерцание во время движения
                if i % 2 == 0:
                    self.simulate_color_change()
                
                time.sleep(0.03)
            
            pyautogui.moveTo(original_x, original_y, duration=0.1)
    
    def cursor_spin(self):
        """Круговое движение с мерцанием цветов"""
        if random.random() < 0.1:
            x, y = pyautogui.position()
            radius = 40
            
            for angle in range(0, 360, 20):
                # Меняем цвет на каждом шаге
                self.simulate_color_change()
                
                new_x = x + radius * math.cos(math.radians(angle))
                new_y = y + radius * math.sin(math.radians(angle))
                pyautogui.moveTo(int(new_x), int(new_y), duration=0.03)
                time.sleep(0.03)
            
            # Финальное мерцание
            for _ in range(3):
                self.simulate_color_change()
            
            pyautogui.moveTo(x, y, duration=0.1)

def rgb_console_effect():
    """Эффект RGB в консоли"""
    effects = [
        "\r🔴⚫🟢 RGB-ПЕРЕКЛЮЧЕНИЕ 🔴⚫🟢",
        "\r🟢🔴⚫ ЦВЕТНАЯ НЕСТАБИЛЬНОСТЬ 🟢🔴⚫",
        "\r⚫🟢🔴 МЕРЦАНИЕ ДИСПЛЕЯ ⚫🟢🔴",
    ]
    
    if random.random() < 0.15:
        print(random.choice(effects) + " " * 20, end='', flush=True)
        time.sleep(0.2)

def fake_errors():
    """Ошибки про RGB"""
    fake_errors_list = [
        "[CRITICAL] RGB channel synchronization failed",
        "[ERROR] Cursor color cycling out of control",
        "[SYSTEM] Red channel overflow detected",
        "[DEBUG] Green pixel corruption: 0x00FF00",
        "[WARNING] Black level too high",
        "[ALERT] RGB spectrum violation",
        "[INFO] Cursor entering RGB mode: 🔴⚫🟢",
        "[ERROR] Color palette corrupted",
        "[SYSTEM] Display driver in RGB panic mode",
        "[CRITICAL] 16.7 million colors error",
        "[DEBUG] Hue shift detected: RED → GREEN",
        "[WARNING] Saturation critical: 150%",
        "[ERROR] Color space conversion failed: sRGB",
        "[SYSTEM] Cursor temperature: 🌈 RAINBOW MODE",
    ]
    
    error_types = ["🔴", "⚫", "🟢", "🌈", "💢", "⚠️"]
    
    print(f"\r{random.choice(error_types)} [{time.strftime('%H:%M:%S')}] {random.choice(fake_errors_list)}" + " " * 40)

def progress_bar(seconds_passed, total_seconds=60):
    """RGB прогресс-бар"""
    percent = int((seconds_passed / total_seconds) * 100)
    filled = int(percent / 5)
    
    # RGB прогресс-бар
    bar_parts = []
    for i in range(filled):
        if i % 3 == 0:
            bar_parts.append("🟥")  # Красный
        elif i % 3 == 1:
            bar_parts.append("⬛")  # Чёрный
        else:
            bar_parts.append("🟩")  # Зелёный
    
    bar = "".join(bar_parts) + "⬜" * (20 - filled)
    
    remaining = total_seconds - seconds_passed
    time_str = f"{int(remaining//60):02d}:{int(remaining%60):02d}"
    
    # Текущий цвет для текста
    color_idx = int(time.time() * 2) % 3
    color_dot = ["🔴", "⚫", "🟢"][color_idx]
    
    print(f"\r{color_dot} RGB-режим: |{bar}| {percent}% | осталось: {time_str} {color_dot}", end="", flush=True)

def show_rgb_show():
    """Показывает RGB-представление"""
    if random.random() < 0.05:  # 5% шанс
        frames = [
            """
    ╔════════════════════════════════════╗
    ║       🔴⚫🟢 RGB ШОУ! 🔴⚫🟢      ║
    ║     КУРСОР МЕНЯЕТ ЦВЕТА!           ║
    ║     КРАСНЫЙ → ЧЁРНЫЙ → ЗЕЛЁНЫЙ     ║
    ║        ⚡ МЕРЦАНИЕ ⚡               ║
    ╚════════════════════════════════════╝
            """,
            """
    ╔════════════════════════════════════╗
    ║       🌈 RGB MODE ACTIVATED 🌈     ║
    ║    COLOR CYCLING: 3Hz              ║
    ║    🔴 ⚫ 🟢 🔴 ⚫ 🟢               ║
    ║    DISPLAY CALIBRATION ERROR       ║
    ╚════════════════════════════════════╝
            """
        ]
        print(random.choice(frames))

def main():
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 60)
    print("🌈 RGB КУРСОР - РОЗЫГРЫШ 🌈")
    print("=" * 60)
    print("\n🚀 Запуск через 3 секунды...")
    print("📌 Информация:")
    print("  • Курсор будет МЕРЦАТЬ: 🔴 КРАСНЫЙ → ⚫ ЧЁРНЫЙ → 🟢 ЗЕЛЁНЫЙ")
    print("  • Будет временно дёргаться")
    print("  • Клавиатура работает нормально")
    print("  • RGB-эффекты в консоли")
    print("  • Скрипт сам остановится через 60 секунд")
    print("  • Ctrl+C для досрочного выхода")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    os.system('clear' if os.name == 'posix' else 'cls')
    
    joker = MouseJoker()
    
    print("🔄 Активация RGB режима...")
    time.sleep(1)
    
    # Запускаем движения мыши
    mouse_thread = threading.Thread(target=joker.move_mouse_randomly, daemon=True)
    mouse_thread.start()
    
    start_time = time.time()
    error_counter = 0
    last_error_time = 0
    last_sound_time = 0
    last_color_change = 0
    
    try:
        while time.time() - start_time < 60:
            current_time = time.time()
            elapsed = current_time - start_time
            
            progress_bar(elapsed)
            
            # Меняем цвет каждые 2 секунды
            if current_time - last_color_change > 2:
                joker.simulate_color_change()
                last_color_change = current_time
            
            # RGB вспышки
            joker.rgb_flash_effect()
            
            # Движения курсора
            if random.random() < 0.3:
                joker.make_cursor_jumpy()
            
            if random.random() < 0.05:
                joker.cursor_spin()
            
            # Ошибки про RGB
            if current_time - last_error_time > 1.5 and random.random() < 0.5:
                fake_errors()
                error_counter += 1
                last_error_time = current_time
            
            # RGB эффект в консоли
            rgb_console_effect()
            
            # Шоу
            show_rgb_show()
            
            # Звуки
            if current_time - last_sound_time > 4 and random.random() < 0.2:
                print('\a', end='', flush=True)
                last_sound_time = current_time
            
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n\n✨ Ручная остановка!")
    finally:
        joker.joke_active = False
        time.sleep(0.5)
        
        print("\n" + "="*60)
        print("🎉 RGB-ШОУ ОКОНЧЕНО! 🎉")
        print("="*60)
        print(f"\n✅ Курсор恢复正常!")
        print(f"📊 Статистика:")
        print(f"  • Создано RGB-ошибок: {error_counter}")
        print(f"  • Время работы: {int(time.time() - start_time)} секунд")
        print("\n😊 Цвета восстановлены!")
        
        try:
            screen_width, screen_height = pyautogui.size()
            pyautogui.moveTo(screen_width//2, screen_height//2, duration=0.5)
            print("🖱️ Курсор возвращён в центр экрана")
        except:
            pass
        
        print("\nНажмите Enter для выхода...")
        input()

if __name__ == "__main__":
    main()
