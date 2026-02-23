#!/usr/bin/env python3
import os
import shutil
import psutil
import subprocess
import readline
import sys
from datetime import datetime
import glob

class TermOS:
    def __init__(self):
        self.username = "user"
        try:
            self.username = os.getlogin()
        except:
            self.username = os.environ.get('USER', 'user')
        
        self.current_dir = os.path.expanduser("~")
        self.trash_dir = os.path.expanduser("~/.trash_os")
        self.python_env = os.path.expanduser("~/termos_python_env")
        self.running = True
        
        # Расширенная цветовая гамма для Termux
        self.CYAN = '\033[96m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.MAGENTA = '\033[95m'
        self.RED = '\033[91m'
        self.PURPLE = '\033[35m'
        self.ORANGE = '\033[38;5;214m'
        self.PINK = '\033[38;5;206m'
        self.TEAL = '\033[38;5;37m'
        self.LIME = '\033[38;5;154m'
        self.GOLD = '\033[38;5;220m'
        self.SILVER = '\033[38;5;250m'
        self.RESET = '\033[0m'
        
        # Стили текста
        self.BOLD = '\033[1m'
        self.ITALIC = '\033[3m'
        self.UNDERLINE = '\033[4m'
        self.BLINK = '\033[5m'
        self.REVERSE = '\033[7m'
        self.STRIKE = '\033[9m'
        
        # Комбинированные стили
        self.BOLD_ITALIC = '\033[1;3m'
        self.BOLD_UNDERLINE = '\033[1;4m'
        self.ITALIC_UNDERLINE = '\033[3;4m'
        self.BOLD_ITALIC_UNDERLINE = '\033[1;3;4m'
        
        self.create_trash_dir()
        self.create_python_env()
        
        self.commands = {
            'help': self.show_help,
            'ls': self.list_files,
            'cd': self.change_directory,
            'pwd': self.show_current_dir,
            'mkdir': self.make_directory,
            'rmdir': self.remove_directory,
            'rm': self.remove_file,
            'cp': self.copy_item,
            'mv': self.move_item,
            'trash': self.show_trash,
            'restore': self.restore_from_trash,
            'emptytrash': self.empty_trash,
            'sysinfo': self.system_info,
            'memory': self.show_memory,
            'clear': self.clear_screen,
            'exit': self.exit_os,
            'touch': self.create_file,
            'cat': self.show_file_content,
            'edit': self.edit_file,
            'find': self.find_files,
            'nano': self.run_nano,
            'python': self.run_python,
            'python3': self.run_python,
            'pip': self.run_pip,
            'pip3': self.run_pip,
            'venv': self.create_venv,
            'activate': self.activate_venv,
            'pkg': self.run_pkg_command,
            'apt': self.run_pkg_command,
            'bash': self.run_bash,
            'termux-open': self.open_termux_file,
            'ps': self.show_processes,
            'kill': self.kill_process,
            'ifconfig': self.show_network,
            'date': self.show_date,
            'whoami': self.show_user
        }
    
    def create_trash_dir(self):
        """Создание директории корзины"""
        if not os.path.exists(self.trash_dir):
            os.makedirs(self.trash_dir)
            print(f"{self.GREEN}✓{self.RESET} {self.ITALIC}🗑️ Корзина создана{self.RESET} 🎉")
    
    def create_python_env(self):
        """Создание директории для Python проектов"""
        if not os.path.exists(self.python_env):
            os.makedirs(self.python_env)
            example_script = os.path.join(self.python_env, "example.py")
            with open(example_script, 'w') as f:
                f.write('''#!/usr/bin/env python3
def main():
    print("╔════════════════════════════════════════╗")
    print("║    🐍 Добро пожаловать в Python в TermOS 🐍 ║")
    print("╚════════════════════════════════════════╝")
    
    name = input("👤 Как вас зовут? ")
    print(f"✨ Приятно познакомиться, {name}! ✨")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"📊 Текущее использование RAM: {memory.percent}%")
    except:
        print("📱 TermOS работает в Termux!")

if __name__ == "__main__":
    main()
''')
            os.chmod(example_script, 0o755)
            print(f"{self.GREEN}✓{self.RESET} {self.ITALIC}🐍 Python окружение создано{self.RESET} 🚀")
    
    def show_help(self, args):
        """Показать справку с расширенным стилем и смайликами"""
        print(f"\n{self.BOLD}{self.CYAN}╔══════════════════════════════════════════════════════════╗{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}        {self.BOLD_ITALIC}{self.GOLD}🤖 TermOS для Termux - Полная справка 🤖{self.RESET}        {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
        
        # Управление файлами
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.GREEN}📁 УПРАВЛЕНИЕ ФАЙЛАМИ:{self.RESET}{' ' * 34}{self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}ls [путь]{self.RESET}     - 👀 показать содержимое                     {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}cd [папка]{self.RESET}    - 🚶 перейти в папку                         {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}pwd{self.RESET}           - 📍 текущий путь                            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}mkdir [имя]{self.RESET}   - 📂 создать папку                           {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}rmdir [имя]{self.RESET}   - 🗑️ удалить папку                           {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}rm [файл]{self.RESET}     - 🗑️ удалить файл (в корзину)                {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}cp [ист] [цель]{self.RESET} - 📋 копировать                            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}mv [ист] [цель]{self.RESET} - 📦 переместить                           {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}touch [файл]{self.RESET}  - ✨ создать файл                            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}cat [файл]{self.RESET}    - 📖 показать содержимое                     {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}edit [файл]{self.RESET}   - ✏️ редактировать файл                      {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.TEAL}find [имя]{self.RESET}    - 🔍 найти файлы                             {self.BOLD}{self.CYAN}║{self.RESET}")
        
        print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
        
        # Корзина
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.ORANGE}🗑️ КОРЗИНА:{self.RESET}{' ' * 41}{self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.ORANGE}trash{self.RESET}         - 🗑️ показать корзину                        {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.ORANGE}restore [файл]{self.RESET}- ♻️ восстановить из корзины                 {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.ORANGE}emptytrash{self.RESET}    - 🧹 очистить корзину                        {self.BOLD}{self.CYAN}║{self.RESET}")
        
        print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
        
        # Python
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.MAGENTA}🐍 PYTHON:{self.RESET}{' ' * 42}{self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.MAGENTA}python [файл]{self.RESET} - 🐍 запустить Python скрипт                 {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.MAGENTA}python{self.RESET}        - 💻 открыть Python интерпретатор            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.MAGENTA}pip [команда]{self.RESET} - 📦 управление Python пакетами              {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.MAGENTA}venv [имя]{self.RESET}    - 🏗️ создать виртуальное окружение           {self.BOLD}{self.CYAN}║{self.RESET}")
        
        print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
        
        # Termux
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.BLUE}📱 TERMUX КОМАНДЫ:{self.RESET}{' ' * 34}{self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.BLUE}pkg [команда]{self.RESET} - 📦 управление пакетами Termux              {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.BLUE}termux-open [файл]{self.RESET} - 📱 открыть файл в Android             {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.BLUE}nano [файл]{self.RESET}   - 📝 редактировать в nano                    {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.BLUE}bash{self.RESET}          - 🐚 открыть Bash shell                       {self.BOLD}{self.CYAN}║{self.RESET}")
        
        print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
        
        # Система
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.PINK}ℹ️ СИСТЕМА:{self.RESET}{' ' * 41}{self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.PINK}memory{self.RESET}        - 💾 показать информацию о памяти           {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.PINK}sysinfo{self.RESET}       - 🤖 информация о системе                   {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.PINK}clear{self.RESET}         - 🧹 очистить экран                         {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.PINK}help{self.RESET}          - ❓ эта справка                            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}  {self.ITALIC}{self.PINK}exit{self.RESET}          - 👋 выход                                  {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}╚══════════════════════════════════════════════════════════╝{self.RESET}")
        print(f"\n{self.BOLD}{self.GOLD}✨ Приятной работы в TermOS! ✨{self.RESET}")
    
    def run_nano(self, args):
        filename = args[0] if args else None
        try:
            cmd = ['nano']
            if filename:
                full_path = os.path.join(self.current_dir, filename)
                dir_path = os.path.dirname(full_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                cmd.append(filename)
            print(f"{self.YELLOW}📝 Запуск nano...{self.RESET}")
            subprocess.run(cmd)
            print(f"{self.GREEN}✓ Редактирование завершено{self.RESET} ✨")
        except FileNotFoundError:
            print(f"{self.RED}❌ nano не установлен. Установите: pkg install nano{self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при запуске nano: {e}{self.RESET}")
    
    def run_python(self, args):
        if args:
            script_path = os.path.join(self.current_dir, args[0])
            if os.path.exists(script_path):
                try:
                    if script_path.endswith('.py'):
                        print(f"{self.MAGENTA}🐍 Запуск Python скрипта...{self.RESET}")
                        subprocess.run(['python', script_path] + args[1:])
                        print(f"{self.GREEN}✓ Скрипт выполнен{self.RESET} ✨")
                    else:
                        print(f"{self.YELLOW}⚠️ Файл {args[0]} не является Python скриптом{self.RESET}")
                except Exception as e:
                    print(f"{self.RED}❌ Ошибка при выполнении скрипта: {e}{self.RESET}")
            else:
                print(f"{self.RED}❌ Файл {args[0]} не найден{self.RESET}")
                self.show_similar_files(args[0])
        else:
            print(f"{self.GREEN}🐍 Python {sys.version}{self.RESET}")
            print(f"{self.YELLOW}💡 Type 'exit()' to return to TermOS{self.RESET}")
            try:
                subprocess.run(['python'])
            except Exception as e:
                print(f"{self.RED}❌ Ошибка при запуске Python: {e}{self.RESET}")
    
    def create_venv(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя виртуального окружения{self.RESET}")
            return
        
        venv_name = args[0]
        venv_path = os.path.join(self.current_dir, venv_name)
        
        try:
            print(f"{self.MAGENTA}🏗️ Создание виртуального окружения {venv_name}...{self.RESET}")
            subprocess.run(['python', '-m', 'venv', venv_path])
            print(f"{self.GREEN}✓ Виртуальное окружение '{venv_name}' создано{self.RESET} 🎉")
            print(f"{self.CYAN}  🔌 Активируйте: source {venv_name}/bin/activate{self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при создании venv: {e}{self.RESET}")
    
    def activate_venv(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя виртуального окружения{self.RESET}")
            return
        
        venv_name = args[0]
        venv_path = os.path.join(self.current_dir, venv_name)
        
        if os.path.exists(venv_path):
            print(f"\n{self.GREEN}🔌 Для активации виртуального окружения '{venv_name}':{self.RESET}")
            print(f"{self.CYAN}  ✨ source {venv_name}/bin/activate{self.RESET}")
            print(f"\n{self.YELLOW}🔌 Для деактивации:{self.RESET}")
            print(f"{self.CYAN}  ✨ deactivate{self.RESET}")
        else:
            print(f"{self.RED}❌ Виртуальное окружение '{venv_name}' не найдено{self.RESET}")
    
    def show_similar_files(self, pattern):
        try:
            files = os.listdir(self.current_dir)
            similar = [f for f in files if pattern.lower() in f.lower()]
            if similar:
                print(f"\n{self.BOLD}{self.CYAN}🔍 Похожие файлы:{self.RESET}")
                for f in similar[:5]:
                    if f.endswith('.py'):
                        print(f"  {self.MAGENTA}🐍 {self.BOLD}{f}{self.RESET}")
                    elif os.path.isdir(os.path.join(self.current_dir, f)):
                        print(f"  {self.BLUE}📁 {self.BOLD}{f}{self.RESET}")
                    else:
                        print(f"  📄 {self.ITALIC}{f}{self.RESET}")
        except:
            pass
    
    def run_pip(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Использование: pip install <пакет> | pip list | pip uninstall <пакет>{self.RESET}")
            return
        
        try:
            cmd = ['pip'] + args
            print(f"{self.MAGENTA}📦 Выполнение pip...{self.RESET}")
            subprocess.run(cmd)
            print(f"{self.GREEN}✓ Готово{self.RESET} ✨")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при выполнении pip: {e}{self.RESET}")
    
    def run_pkg_command(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Использование: pkg install <пакет> | pkg update | pkg upgrade{self.RESET}")
            return
        
        try:
            cmd = ['pkg'] + args
            print(f"{self.BLUE}📦 Выполнение pkg...{self.RESET}")
            subprocess.run(cmd)
            print(f"{self.GREEN}✓ Готово{self.RESET} ✨")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при выполнении pkg: {e}{self.RESET}")
    
    def open_termux_file(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите файл для открытия{self.RESET}")
            return
        
        file_path = os.path.join(self.current_dir, args[0])
        if os.path.exists(file_path):
            try:
                print(f"{self.BLUE}📱 Открытие файла в Android...{self.RESET}")
                subprocess.run(['termux-open', file_path])
                print(f"{self.GREEN}✓ Файл открыт{self.RESET} 📱")
            except:
                print(f"{self.RED}❌ termux-open не доступен{self.RESET}")
        else:
            print(f"{self.RED}❌ Файл не найден{self.RESET}")
    
    def run_bash(self, args):
        try:
            print(f"{self.YELLOW}🐚 Запуск Bash shell (exit для возврата в TermOS){self.RESET}")
            subprocess.run(['bash'])
            print(f"{self.GREEN}✓ Возврат в TermOS{self.RESET} 👋")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при запуске bash: {e}{self.RESET}")
    
    def show_processes(self, args):
        try:
            print(f"{self.CYAN}📊 Список процессов:{self.RESET}")
            subprocess.run(['ps', 'aux'])
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def kill_process(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите PID процесса{self.RESET}")
            return
        
        try:
            pid = int(args[0])
            os.kill(pid, 15)
            print(f"{self.GREEN}✓ Процесс {pid} завершен{self.RESET} 💀")
        except ValueError:
            print(f"{self.RED}❌ Некорректный PID{self.RESET}")
        except ProcessLookupError:
            print(f"{self.RED}❌ Процесс {pid} не найден{self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def show_network(self, args):
        try:
            print(f"{self.CYAN}🌐 Сетевые интерфейсы:{self.RESET}")
            subprocess.run(['ifconfig'])
        except:
            try:
                subprocess.run(['ip', 'addr'])
            except:
                print(f"{self.YELLOW}⚠️ Команда ifconfig не найдена{self.RESET}")
    
    def show_date(self, args):
        now = datetime.now()
        print(f"{self.CYAN}📅 {now.strftime('%Y-%m-%d %H:%M:%S')}{self.RESET} ⏰")
    
    def show_user(self, args):
        print(f"{self.GREEN}👤 {self.username}{self.RESET}")
    
    def list_files(self, args):
        path = args[0] if args else self.current_dir
        try:
            items = os.listdir(path)
            print(f"\n{self.BOLD}{self.BLUE}📂 Содержимое {path}:{self.RESET}")
            print(f"{self.ITALIC}{self.SILVER}════════════════════════════════════════════════════════{self.RESET}")
            
            dirs = []
            files = []
            for item in sorted(items):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    dirs.append(item)
                else:
                    files.append(item)
            
            for i, item in enumerate(dirs):
                if i % 3 == 0:
                    color = self.BLUE
                elif i % 3 == 1:
                    color = self.TEAL
                else:
                    color = self.CYAN
                print(f"  {color}📁 {self.BOLD}{item}/{self.RESET}")
            
            for item in files:
                full_path = os.path.join(path, item)
                try:
                    size = os.path.getsize(full_path)
                except:
                    size = 0
                
                if item.endswith(('.py', '.py3')):
                    print(f"  {self.MAGENTA}🐍 {self.BOLD}{item}{self.RESET} {self.ITALIC}({self.format_size(size)}){self.RESET}")
                elif item.endswith(('.txt', '.md')):
                    print(f"  {self.YELLOW}📄 {self.ITALIC}{item}{self.RESET} ({self.format_size(size)})")
                elif item.endswith(('.json', '.yml', '.yaml')):
                    print(f"  {self.ORANGE}⚙️ {self.ITALIC}{item}{self.RESET} ({self.format_size(size)})")
                elif item.endswith(('.sh', '.bash')):
                    print(f"  {self.LIME}⚡ {self.BOLD}{item}{self.RESET} ({self.format_size(size)})")
                elif item.endswith(('.jpg', '.png', '.gif', '.jpeg')):
                    print(f"  {self.PINK}🖼️ {item}{self.RESET} ({self.format_size(size)})")
                elif item.endswith(('.mp3', '.wav', '.ogg')):
                    print(f"  {self.PURPLE}🎵 {item}{self.RESET} ({self.format_size(size)})")
                elif item.endswith(('.mp4', '.avi', '.mkv')):
                    print(f"  {self.RED}🎬 {item}{self.RESET} ({self.format_size(size)})")
                elif item.endswith(('.zip', '.tar', '.gz')):
                    print(f"  {self.GOLD}📦 {item}{self.RESET} ({self.format_size(size)})")
                elif item.startswith('.'):
                    print(f"  {self.SILVER}🔒 {self.ITALIC}{item}{self.RESET} ({self.format_size(size)})")
                else:
                    print(f"  📄 {item} ({self.format_size(size)})")
            
            print(f"\n{self.BOLD}{self.GREEN}📊 Статистика:{self.RESET} {self.BOLD}{len(dirs)}{self.RESET} 📁 папок, {self.BOLD}{len(files)}{self.RESET} 📄 файлов")
            
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def change_directory(self, args):
        if not args:
            self.current_dir = os.path.expanduser("~")
            print(f"{self.GREEN}📍 Перешли в домашнюю директорию{self.RESET} 🏠")
        else:
            new_path = args[0]
            if new_path == "..":
                self.current_dir = os.path.dirname(self.current_dir)
                print(f"{self.GREEN}📍 Назад: {self.current_dir}{self.RESET} ⬆️")
            elif new_path.startswith("/"):
                if os.path.exists(new_path) and os.path.isdir(new_path):
                    self.current_dir = new_path
                    print(f"{self.GREEN}📍 Перешли в {new_path}{self.RESET} 🚶")
                else:
                    print(f"{self.RED}❌ Путь не существует{self.RESET}")
                    return
            else:
                full_path = os.path.join(self.current_dir, new_path)
                if os.path.exists(full_path) and os.path.isdir(full_path):
                    self.current_dir = full_path
                    print(f"{self.GREEN}📍 Перешли в {new_path}{self.RESET} 🚶")
                else:
                    print(f"{self.RED}❌ Директория не существует{self.RESET}")
                    return
    
    def show_current_dir(self, args):
        print(f"{self.CYAN}📍 {self.current_dir}{self.RESET}")
    
    def make_directory(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя папки{self.RESET}")
            return
        path = os.path.join(self.current_dir, args[0])
        try:
            os.makedirs(path)
            print(f"{self.GREEN}✓ Папка {args[0]} создана{self.RESET} 📂✨")
        except FileExistsError:
            print(f"{self.YELLOW}⚠️ Папка {args[0]} уже существует{self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def remove_directory(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя папки{self.RESET}")
            return
        path = os.path.join(self.current_dir, args[0])
        try:
            if os.path.exists(path) and os.path.isdir(path):
                trash_path = os.path.join(self.trash_dir, args[0])
                if os.path.exists(trash_path):
                    base, ext = os.path.splitext(args[0])
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    trash_path = os.path.join(self.trash_dir, f"{base}_{timestamp}{ext}")
                
                shutil.move(path, trash_path)
                print(f"{self.GREEN}✓ Папка {args[0]} перемещена в корзину{self.RESET} 🗑️")
            else:
                print(f"{self.RED}❌ Папка не найдена{self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def remove_file(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя файла{self.RESET}")
            return
        
        file_path = os.path.join(self.current_dir, args[0])
        if os.path.exists(file_path) and os.path.isfile(file_path):
            trash_path = os.path.join(self.trash_dir, args[0])
            if os.path.exists(trash_path):
                base, ext = os.path.splitext(args[0])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                trash_path = os.path.join(self.trash_dir, f"{base}_{timestamp}{ext}")
            
            shutil.move(file_path, trash_path)
            print(f"{self.GREEN}✓ Файл {args[0]} перемещен в корзину{self.RESET} 🗑️")
        else:
            print(f"{self.RED}❌ Файл не найден{self.RESET}")
            self.show_similar_files(args[0])
    
    def copy_item(self, args):
        if len(args) < 2:
            print(f"{self.YELLOW}⚠️ Укажите источник и цель{self.RESET}")
            return
        
        src = args[0]
        dst = args[1]
        
        src_path = os.path.join(self.current_dir, src)
        dst_path = os.path.join(self.current_dir, dst)
        
        try:
            if not os.path.exists(src_path):
                print(f"{self.RED}❌ Источник '{src}' не найден{self.RESET}")
                self.show_similar_files(src)
                return
            
            if os.path.isdir(dst_path):
                dst_path = os.path.join(dst_path, os.path.basename(src_path))
            
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
                print(f"{self.GREEN}✓ Папка скопирована{self.RESET} 📂✨")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"{self.GREEN}✓ Файл скопирован{self.RESET} 📄✨")
            
            print(f"{self.GREEN}  {src} -> {os.path.relpath(dst_path, self.current_dir)}{self.RESET}")
            
        except FileExistsError:
            print(f"{self.YELLOW}⚠️ Цель '{dst}' уже существует{self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при копировании: {e}{self.RESET}")
    
    def move_item(self, args):
        if len(args) < 2:
            print(f"{self.YELLOW}⚠️ Укажите источник и цель{self.RESET}")
            return
        
        src = args[0]
        dst = args[1]
        
        src_path = os.path.join(self.current_dir, src)
        dst_path = os.path.join(self.current_dir, dst)
        
        try:
            if not os.path.exists(src_path):
                print(f"{self.RED}❌ Источник '{src}' не найден{self.RESET}")
                print(f"{self.YELLOW}   📍 Текущая директория: {self.current_dir}{self.RESET}")
                self.show_similar_files(src)
                return
            
            if os.path.isdir(dst_path):
                dst_path = os.path.join(dst_path, os.path.basename(src_path))
                print(f"{self.CYAN}📁 Перемещение в папку: {dst}{self.RESET}")
            
            if os.path.exists(dst_path) and not os.path.isdir(dst_path):
                confirm = input(f"{self.YELLOW}⚠️ Файл '{os.path.basename(dst_path)}' уже существует. Перезаписать? (y/n): {self.RESET}")
                if confirm.lower() != 'y':
                    print(f"{self.YELLOW}⏸️ Операция отменена{self.RESET}")
                    return
            
            shutil.move(src_path, dst_path)
            rel_dst = os.path.relpath(dst_path, self.current_dir)
            
            if os.path.isdir(dst_path):
                print(f"{self.GREEN}✓ Папка перемещена{self.RESET} 📂➡️📁")
            else:
                print(f"{self.GREEN}✓ Файл перемещен{self.RESET} 📄➡️📁")
            print(f"{self.GREEN}  {src} -> {rel_dst}{self.RESET}")
            
        except Exception as e:
            print(f"{self.RED}❌ Ошибка при перемещении: {e}{self.RESET}")
    
    def show_trash(self, args):
        items = os.listdir(self.trash_dir)
        if items:
            print(f"\n{self.BOLD}{self.ORANGE}🗑️ Корзина содержит:{self.RESET}")
            print(f"{self.ITALIC}{self.SILVER}══════════════════════════════════════════════════════════════{self.RESET}")
            
            total_size = 0
            for i, item in enumerate(sorted(items), 1):
                path = os.path.join(self.trash_dir, item)
                mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                
                if os.path.isdir(path):
                    size = self.get_dir_size(path)
                    total_size += size
                    print(f"  {self.BOLD}{i}.{self.RESET} {self.BLUE}📁 {self.BOLD}{item}{self.RESET} {self.ITALIC}({self.format_size(size)}){self.RESET}")
                else:
                    size = os.path.getsize(path)
                    total_size += size
                    if item.endswith('.py'):
                        print(f"  {self.BOLD}{i}.{self.RESET} {self.MAGENTA}🐍 {self.BOLD}{item}{self.RESET} {self.ITALIC}({self.format_size(size)}){self.RESET}")
                    elif item.endswith(('.jpg', '.png', '.gif')):
                        print(f"  {self.BOLD}{i}.{self.RESET} {self.PINK}🖼️ {self.ITALIC}{item}{self.RESET} ({self.format_size(size)})")
                    elif item.endswith(('.mp3', '.wav')):
                        print(f"  {self.BOLD}{i}.{self.RESET} {self.PURPLE}🎵 {self.ITALIC}{item}{self.RESET} ({self.format_size(size)})")
                    else:
                        print(f"  {self.BOLD}{i}.{self.RESET} 📄 {self.ITALIC}{item}{self.RESET} ({self.format_size(size)})")
                
                print(f"     {self.SILVER}⏰ удален: {mod_time.strftime('%Y-%m-%d %H:%M')}{self.RESET}")
            
            print(f"\n{self.BOLD}📊 Итого:{self.RESET} {len(items)} элементов, {self.BOLD}{self.ORANGE}{self.format_size(total_size)}{self.RESET}")
        else:
            print(f"{self.ITALIC}{self.YELLOW}🗑️ Корзина пуста ✨{self.RESET}")
    
    def get_dir_size(self, path):
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except:
                    pass
        return total
    
    def restore_from_trash(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя файла для восстановления{self.RESET}")
            return
        
        found = []
        for item in os.listdir(self.trash_dir):
            if args[0] in item:
                found.append(item)
        
        if len(found) == 0:
            print(f"{self.RED}❌ Файл не найден в корзине{self.RESET}")
            return
        elif len(found) > 1:
            print(f"{self.YELLOW}🔍 Найдено несколько файлов:{self.RESET}")
            for i, item in enumerate(found, 1):
                print(f"  {i}. {item}")
            try:
                choice = int(input(f"{self.CYAN}👉 Выберите номер: {self.RESET}")) - 1
                if 0 <= choice < len(found):
                    trash_path = os.path.join(self.trash_dir, found[choice])
                else:
                    print(f"{self.RED}❌ Неверный выбор{self.RESET}")
                    return
            except:
                print(f"{self.RED}❌ Неверный выбор{self.RESET}")
                return
        else:
            trash_path = os.path.join(self.trash_dir, found[0])
        
        dest_path = os.path.join(self.current_dir, os.path.basename(trash_path))
        if os.path.exists(dest_path):
            print(f"{self.YELLOW}⚠️ Файл {os.path.basename(trash_path)} уже существует в текущей директории{self.RESET}")
            return
        
        shutil.move(trash_path, self.current_dir)
        print(f"{self.GREEN}✓ Файл восстановлен{self.RESET} ♻️✨")
    
    def empty_trash(self, args):
        items = os.listdir(self.trash_dir)
        if not items:
            print(f"{self.YELLOW}🗑️ Корзина уже пуста ✨{self.RESET}")
            return
        
        print(f"{self.RED}⚠️ В корзине {len(items)} элементов{self.RESET}")
        confirm = input(f"{self.RED}🗑️ Очистить корзину? Все файлы будут удалены безвозвратно (y/n): {self.RESET}")
        if confirm.lower() == 'y':
            for item in items:
                path = os.path.join(self.trash_dir, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            print(f"{self.GREEN}✓ Корзина очищена{self.RESET} 🧹✨")
    
    def show_memory(self, args):
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            print(f"\n{self.BOLD}{self.CYAN}╔══════════════════════════════════════════════════════════╗{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}            {self.BOLD_ITALIC}{self.GOLD}💾 ИНФОРМАЦИЯ О ПАМЯТИ 📊{self.RESET}            {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
            
            print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.MAGENTA}💾 ОПЕРАТИВНАЯ ПАМЯТЬ (RAM):{self.RESET}                  {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   Всего:     {self.BOLD}{self.GREEN}{self.format_size(memory.total):>15}{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   Доступно:  {self.BOLD}{self.LIME}{self.format_size(memory.available):>15}{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   Загрузка:  {self.BOLD}{self.ORANGE}{memory.percent:>13.1f}%{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
            
            bar_length = 30
            filled = int(bar_length * memory.percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   [{self.RED if memory.percent > 80 else self.YELLOW if memory.percent > 50 else self.GREEN}{bar}{self.RESET}] {self.BOLD}{memory.percent:.1f}%{self.RESET}         {self.BOLD}{self.CYAN}║{self.RESET}")
            
            print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
            
            print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}{self.BLUE}💿 ДИСКОВОЕ ПРОСТРАНСТВО:{self.RESET}                         {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   Всего:     {self.BOLD}{self.GREEN}{self.format_size(disk.total):>15}{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   Свободно:  {self.BOLD}{self.LIME}{self.format_size(disk.free):>15}{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   Загрузка:  {self.BOLD}{self.ORANGE}{disk.percent:>13.1f}%{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
            
            filled = int(bar_length * disk.percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"{self.BOLD}{self.CYAN}║{self.RESET}   [{self.RED if disk.percent > 80 else self.YELLOW if disk.percent > 50 else self.GREEN}{bar}{self.RESET}] {self.BOLD}{disk.percent:.1f}%{self.RESET}         {self.BOLD}{self.CYAN}║{self.RESET}")
            
            print(f"{self.BOLD}{self.CYAN}╚══════════════════════════════════════════════════════════╝{self.RESET}")
        except:
            print(f"{self.YELLOW}{self.ITALIC}⚠️ Информация о памяти недоступна в Termux{self.RESET}")
    
    def system_info(self, args):
        print(f"\n{self.BOLD}{self.CYAN}╔══════════════════════════════════════════════════════════╗{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}            {self.BOLD_ITALIC}{self.GOLD}🤖 ИНФОРМАЦИЯ О СИСТЕМЕ 📱{self.RESET}            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}╠══════════════════════════════════════════════════════════╣{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}👤 Пользователь:{self.RESET} {self.GREEN}{self.username:<30}{self.RESET} {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}📱 Платформа:{self.RESET} {self.MAGENTA}{self.BOLD}Termux{self.RESET}{' ' * 35}{self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}🐍 Python:{self.RESET} {self.ORANGE}{sys.version.split()[0]:<30}{self.RESET} {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}📁 Текущая директория:{self.RESET} {self.CYAN}{self.ITALIC}{os.path.basename(self.current_dir):<18}{self.RESET} {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET} {self.BOLD}🕒 Время:{self.RESET} {self.LIME}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{self.RESET}          {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}╚══════════════════════════════════════════════════════════╝{self.RESET}")
    
    def create_file(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя файла{self.RESET}")
            return
        path = os.path.join(self.current_dir, args[0])
        try:
            with open(path, 'a'):
                os.utime(path, None)
            print(f"{self.GREEN}✓ Файл {args[0]} создан{self.RESET} ✨📄")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def show_file_content(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя файла{self.RESET}")
            return
        path = os.path.join(self.current_dir, args[0])
        try:
            if not os.path.exists(path):
                print(f"{self.RED}❌ Файл не найден{self.RESET}")
                self.show_similar_files(args[0])
                return
            
            with open(path, 'r') as f:
                content = f.read()
                print(f"\n{self.YELLOW}📄 Содержимое {args[0]}:{self.RESET}")
                print(f"{self.CYAN}════════════════════════════════════════════════════════{self.RESET}")
                print(content)
                if not content.endswith('\n'):
                    print()
        except UnicodeDecodeError:
            print(f"{self.YELLOW}⚠️ Невозможно прочитать файл (возможно, бинарный){self.RESET}")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def edit_file(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя файла{self.RESET}")
            return
        path = os.path.join(self.current_dir, args[0])
        try:
            print(f"{self.GREEN}📝 Редактор (введите текст, Ctrl+D для сохранения и выхода):{self.RESET}")
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            with open(path, 'w') as f:
                f.write('\n'.join(lines))
            print(f"{self.GREEN}✓ Файл сохранен{self.RESET} 💾✨")
        except Exception as e:
            print(f"{self.RED}❌ Ошибка: {e}{self.RESET}")
    
    def find_files(self, args):
        if not args:
            print(f"{self.YELLOW}⚠️ Укажите имя для поиска{self.RESET}")
            return
        pattern = args[0]
        found = []
        
        print(f"{self.CYAN}🔍 Поиск файлов...{self.RESET}")
        for root, dirs, files in os.walk(self.current_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if pattern.lower() in file.lower():
                    full_path = os.path.join(root, file)
                    found.append(full_path)
        
        if found:
            print(f"\n{self.GREEN}🔍 Найдено {len(found)} файлов:{self.RESET} 🎯")
            print(f"{self.CYAN}════════════════════════════════════════════════════════{self.RESET}")
            for i, f in enumerate(found[:20], 1):
                rel_path = os.path.relpath(f, self.current_dir)
                if f.endswith('.py'):
                    print(f"  {i}. {self.MAGENTA}🐍 ./{rel_path}{self.RESET}")
                elif os.path.isdir(f):
                    print(f"  {i}. {self.BLUE}📁 ./{rel_path}{self.RESET}")
                else:
                    print(f"  {i}. 📄 ./{rel_path}")
            if len(found) > 20:
                print(f"  ... и еще {len(found) - 20} файлов")
        else:
            print(f"{self.YELLOW}🔍 Файлы не найдены{self.RESET} 😢")
    
    def clear_screen(self, args):
        os.system('clear')
        self.show_banner()
    
    def show_banner(self):
        print(f"\n{self.BOLD}{self.CYAN}╔══════════════════════════════════════════════════════════╗{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}        {self.BOLD_ITALIC}{self.GOLD}🤖 TermOS для Termux v2.0 🚀{self.RESET}                {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}         {self.ITALIC}{self.LIME}✨ Ваша персональная ОС на Python ✨{self.RESET}            {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{self.RESET}              {self.ITALIC}{self.MAGENTA}📱 Введите 'help' для списка команд 📱{self.RESET}      {self.BOLD}{self.CYAN}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}╚══════════════════════════════════════════════════════════╝{self.RESET}\n")
    
    def exit_os(self, args):
        print(f"\n{self.GREEN}👋 Завершение работы TermOS...{self.RESET} 💤")
        self.running = False
        return False
    
    def format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def run_command(self, command):
        if not command.strip():
            return True
        
        parts = command.strip().split()
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in ['ls', 'grep', 'cat', 'echo', 'which', 'pwd', 'date', 'whoami']:
            try:
                subprocess.run(command, shell=True)
            except Exception as e:
                print(f"{self.RED}❌ Ошибка выполнения: {e}{self.RESET}")
            return True
        
        if cmd in self.commands:
            result = self.commands[cmd](args)
            return result is not False
        else:
            try:
                subprocess.run(command, shell=True)
            except Exception as e:
                print(f"{self.RED}❌ Команда не найдена: {cmd}. Введите 'help' для справки{self.RESET}")
            return True
    
    def run(self):
        self.clear_screen([])
        
        while self.running:
            try:
                home = os.path.expanduser('~')
                if self.current_dir.startswith(home):
                    short_path = '~' + self.current_dir[len(home):]
                else:
                    short_path = self.current_dir
                
                if len(short_path) > 30:
                    short_path = '...' + short_path[-27:]
                
                prompt = f"{self.BOLD}{self.GREEN}{self.username}{self.RESET}{self.BOLD}@{self.RESET}{self.BOLD}{self.MAGENTA}TermOS{self.RESET}:{self.BOLD}{self.BLUE}{short_path}{self.RESET}{self.BOLD}{self.GOLD}$ {self.RESET}"
                command = input(prompt)
                self.run_command(command)
                
            except KeyboardInterrupt:
                print(f"\n{self.ITALIC}{self.YELLOW}⌨️ Используйте 'exit' для выхода{self.RESET}")
            except EOFError:
                print(f"\n{self.BOLD}{self.GREEN}👋 Выход...{self.RESET} 💤")
                break
            except Exception as e:
                print(f"{self.BOLD}{self.RED}⚠️ Ошибка: {e}{self.RESET}")
if __name__ == "__main__":
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    RESET = '\033[0m'
    
    print(f"{BOLD}{GREEN}╔════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{GREEN}║{RESET}    🚀 Запуск TermOS для Termux...    {BOLD}{GREEN}║{RESET}")
    print(f"{BOLD}{GREEN}║{RESET}       {ITALIC}✨ Добро пожаловать! ✨{RESET}       {BOLD}{GREEN}║{RESET}")
    print(f"{BOLD}{GREEN}╚════════════════════════════════════════╝{RESET}")
    
    try:
        import psutil
        print(f"{BOLD}{GREEN}✓{RESET} psutil {GREEN}загружен{RESET} ✅")
    except ImportError:
        print(f"{BOLD}{YELLOW}ℹ️{RESET} {YELLOW}psutil не установлен. Некоторые функции могут не работать.{RESET}")
        print(f"{YELLOW}   Установите: pip install psutil{RESET}")
    
    os_ = TermOS()
    os_.run()
