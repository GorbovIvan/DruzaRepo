import time
import sys
import json
import os

def load_config():
    config_file = "config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default_config
    else:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        return default_config

config = load_config()
device_type = config.get("device", "pc")
separator_config = config["separator"].get(device_type, {"symbol": "#", "count": 40})
separator_symbol = separator_config["symbol"]
separator_count = separator_config["count"]

def slow_print(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def slow_input(prompt, delay=0.03):
    slow_print(prompt, delay)
    return input()

inventory = {
    "equiped": "",
    "pickaxes": []
}

slow_print("Goblin - ролевая игра")
print(separator_symbol * separator_count)

slow_input("Привет, Гоблин! ")
slow_input("Теперь я твой наставник! ")
slow_input("Наставник: осмотрись вокруг, мой новобранец!")
slow_print("*Пещера, на вид гигантская и хранящая в себе множество тайн*")
slow_input("Наставник: это пещера Гоблинберг! Отсюда каждый шахтёр начинает свой путь!")
slow_input("Наставник: держи свою первую кирку! ")
slow_print("*Каменная кирка, немного потрескалась но сойдёт*")
slow_input("Получено: каменная кирка!")

inventory["Каменная кирка"] = 40
inventory["pickaxes"].append("Каменная кирка")

answer = slow_input("Экипировать каменная кирка? д/н ")
if answer == "д":
    slow_print("экипировано")
    inventory["equiped"] = "Каменная кирка"
else:
    slow_print("Экипируете, когда найдете руды")

slow_input("Наставник: в добрый путь, мой ученик!")
print(separator_symbol * separator_count)

answer = slow_input("*Вы идёте по тёмному коридору. Вы видите дверь. Открыть? д/н ")
if answer == 'д':
    slow_print("*Вы видите перед собой сундук. Вы открываете его и находите железную кирку! Вы экипируете её и выходите из комнаты.*")
else:
    slow_print("*Под вами появились шипы и вы от боли уронили свою кирку в пропасть.")
    slow_print("Шипы опустились, вы с облегчением вздохнули и подумали что это знак что нужно войти в таинственную комнату.")
    slow_print("Вы входите в неё и находите в сундуке железную кирку! Вы радостно её экипируете и выходите из комнаты*")

inventory["pickaxes"].append("Железная кирка")
inventory["Железная кирка"] = 150
inventory["equiped"] = "Железная кирка"

slow_input("Нажмите Enter чтобы продолжить...")
slow_print("*Вы сворачиваете по коридору налево но пол под вами резко проваливается.*")
slow_print("*Вы каким то чудом уцелели. Вы открываете глаза после падения и подбираете кирку, что валялась рядом с вами.*")
slow_print("*Вы осматриваетесь и понимаете, что вы находитесь в небольшом руднике*")
slow_print("*Вы подходите к руде и рассматриваете её*")
slow_print("*Гоблинское чутьё подсказывает что это гематит!*")

answer = slow_input(f"Хотите ли вы экипировать другую кирку?(сейчас:{inventory['equiped']}) д/н ")
if answer == "д":
    pick_name = slow_input("Напишите название кирки: ")
    if pick_name in inventory["pickaxes"]:
        if inventory[pick_name] <= 0:
            slow_print(f"Кирка {pick_name} сломана! Нельзя экипировать сломанный инструмент.")
            slow_print("Оставляем текущую кирку.")
        else:
            inventory["equiped"] = pick_name
            slow_print(f"Экипировано {pick_name}")
    else:
        slow_print("Такой кирки нет.")
else:
    slow_print("Эта кирка вас устраивает!")

answer = slow_input("*Добыть? д/н ")
if answer == "д":
    current_pickaxe = inventory["equiped"]

    if not current_pickaxe:
        slow_print("*У вас нет экипированной кирки!*")
    elif inventory[current_pickaxe] <= 0:
        slow_print(f"*Ваша {current_pickaxe} сломана! Нужно экипировать другую кирку.*")
        inventory[current_pickaxe] = 0
    else:
        slow_print("Добыча...")
        time.sleep(5)
        slow_print("*треск!*")
        slow_print("Получено: гематит x3")
        inventory["Гематит"] = 3
        inventory[current_pickaxe] -= 40

        if inventory[current_pickaxe] <= 0:
            slow_print(f"\n*Ваша {current_pickaxe} сломалась от напряжения!*")
            inventory[current_pickaxe] = 0
            slow_print("*Вам нужно экипировать другую кирку, чтобы продолжить добычу.*")
else:
    slow_print("Советую не проходить мимо.")

slow_print("\n--- Состояние инструментов ---")
slow_print(f"Экипировано: {inventory['equiped']}")
for pickaxe in inventory["pickaxes"]:
    status = "СЛОМАНА" if inventory[pickaxe] <= 0 else f"прочность: {inventory[pickaxe]}"
    slow_print(f"{pickaxe}: {status}")
