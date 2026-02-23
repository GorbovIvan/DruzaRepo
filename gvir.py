#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import os
import random
import pyautogui
import threading
import math

# Для работы: pip install pyautogui

class MouseJoker:
    def __init__(self):
        self.joke_active = True
        self.screen_width, self.screen_height = pyautogui.size()
        self.color_cycle = 0
        
    def move_mouse_randomly(self):
        """Агрессивные движения мыши"""
        while self.joke_active:
            current_x, current_y = pyautogui.position()
            
            # ОЧЕНЬ СИЛЬНЫЕ дергания
            new_x = current_x + random.randint(-200, 200)
            new_y = current_y + random.randint(-200, 200)
            
            # Резкие движения
            new_x = max(0, min(new_x, self.screen_width))
            new_y = max(0, min(new_y, self.screen_height))
            
            pyautogui.moveTo(new_x, new_y, duration=0.01)  # ОЧЕНЬ БЫСТРО
            time.sleep(random.uniform(0.01, 0.1))  # Минимальные паузы
    
    def ultra_jumpy(self):
        """Сверх-дергания"""
        if random.random() < 0.8:  # 80% шанс
            original_x, original_y = pyautogui.position()
            jumps = random.randint(15, 30)  # Много прыжков
            
            for _ in range(jumps):
                offset_x = random.randint(-100, 100)
                offset_y = random.randint(-100, 100)
                pyautogui.moveTo(
                    original_x + offset_x, 
                    original_y + offset_y, 
                    duration=0.005  # МИЛЛИСЕКУНДЫ
                )
                time.sleep(0.005)
            
            pyautogui.moveTo(original_x, original_y, duration=0.01)
    
    def chaos_mode(self):
        """Хаотичный режим"""
        if random.random() < 0.3:  # 30% шанс
            for _ in range(random.randint(5, 15)):
                x = random.randint(0, self.screen_width)
                y = random.randint(0, self.screen_height)
                pyautogui.moveTo(x, y, duration=0.001)
                time.sleep(0.001)
    
    def cursor_spin_fast(self):
        """Быстрое кружение"""
        if random.random() < 0.2:  # 20% шанс
            x, y = pyautogui.position()
            radius = random.randint(50, 150)
            
            for angle in range(0, 360, 5):  # Мелкие шаги
                new_x = x + radius * math.cos(math.radians(angle))
                new_y = y + radius * math.sin(math.radians(angle))
                pyautogui.moveTo(int(new_x), int(new_y), duration=0.001)
                time.sleep(0.001)
    
    def vibration_mode(self):
        """Вибрация курсора"""
        if random.random() < 0.5:  # 50% шанс
            x, y = pyautogui.position()
            for _ in range(random.randint(20, 50)):
                pyautogui.moveTo(
                    x + random.randint(-15, 15),
                    y + random.randint(-15, 15),
                    duration=0.001
                )
                time.sleep(0.001)
            pyautogui.moveTo(x, y, duration=0.001)
    
    def teleport_mode(self):
        """Телепортация по экрану"""
        if random.random() < 0.1:  # 10% шанс
            for _ in range(random.randint(3, 8)):
                x = random.randint(0, self.screen_width)
                y = random.randint(0, self.screen_height)
                pyautogui.moveTo(x, y, duration=0)
                time.sleep(0.05)

def annoying_sounds():
    """Звуки (без print)"""
    for _ in range(random.randint(1, 5)):
        sys.stdout.write('\a')
        sys.stdout.flush()
        time.sleep(0.05)

def main():
    # Полностью очищаем экран и отключаем вывод
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Перенаправляем stdout в никуда (подавляем весь вывод)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    joker = MouseJoker()
    
    # Запускаем основное движение
    mouse_thread = threading.Thread(target=joker.move_mouse_randomly, daemon=True)
    mouse_thread.start()
    
    start_time = time.time()
    last_sound_time = 0
    
    try:
        while time.time() - start_time < 60:  # 60 секунд ада
            current_time = time.time()
            
            # МНОГО дерганий
            joker.ultra_jumpy()
            joker.vibration_mode()
            
            # Хаос
            if random.random() < 0.2:
                joker.chaos_mode()
            
            # Кружение
            if random.random() < 0.15:
                joker.cursor_spin_fast()
            
            # Телепортация
            if random.random() < 0.1:
                joker.teleport_mode()
            
            # Звуки (очень часто)
            if current_time - last_sound_time > 0.5:  # Каждые 0.5 секунд
                annoying_sounds()
                last_sound_time = current_time
            
            time.sleep(0.01)  # Микро-паузы
    
    except:
        pass
    finally:
        joker.joke_active = False
        time.sleep(0.1)
        
        # Возвращаем stdout
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        # Возвращаем курсор в центр
        try:
            screen_width, screen_height = pyautogui.size()
            pyautogui.moveTo(screen_width//2, screen_height//2, duration=0.1)
        except:
            pass

if __name__ == "__main__":
    main()
