from tkinter import messagebox, ttk
from scipy import stats
import tkinter as tk
import pandas as pd
import numpy as np
import time


# --- ЧАСТИНА 1: ЛОГІКА ---
class KeyboardAuthLogic:
    def __init__(self):
        self.filename = "etalon.csv"
        self.t_table_student = 0
        self.f_table_fisher = 0
    
    def clean_time_intervals(self, time_intervals):
        alpha = 0.05
        t_T = stats.t.ppf(1 - alpha/2, time_intervals.shape[1])
        print(time_intervals.shape)
        print(time_intervals)
        
        ni = time_intervals.shape[1]
        nj = time_intervals.shape[0]
        
        for i in range(ni):
            time_interval = time_intervals[:, i]
            print(time_interval)
            j = 0
            while j < nj:
                print(i, j)
                tint = np.delete(time_interval, j)
                Mi = tint.mean()
                stdi = tint.std()
                
                t_p = np.abs((time_interval[j] - Mi)/stdi)
                
                if t_p > t_T:
                    time_intervals = np.delete(time_intervals, j, axis=0)
                    nj -= 1
                    break
                j += 1
        print(time_intervals)
        return time_intervals

    def save_template(self, attempts_data):
        print("Файл записано")
        time_intervals = np.array(attempts_data)
        time_intervals = self.clean_time_intervals(time_intervals)
        
        ti_df = pd.DataFrame(time_intervals)
        ti_df.to_csv(self.filename, index=False, header=False)
        
    def load_template(self):
        return pd.read_csv(self.filename).to_numpy()
        

    def verify_user(self, delays, template):
        delays = np.array(delays)
        
        F = 0
        T = 0
        
        n = template.shape[1]
        
        for i in range(template.shape[0]):
            for j in range(delays.shape[0]):
                print(f"template : {template[i]}")
                print(f"delays : {delays[j]}")
                S2max = max(template[i].var(), delays[j].var())
                S2min = min(template[i].var(), delays[j].var())
                
                F_p = S2max/S2min
                
                if F_p > self.f_table_fisher:
                    print("Дисперсії неоднорідні, скіпаємо")
                    F += 1
                else:
                    print("Дисперсії однорідні")
                    S = np.sqrt(((template[i].var() + delays[j].var())*(n-1))/(2*n - 1))
                    
                    t_p = np.abs(template[i].mean() + delays[j].mean())/(S*np.sqrt(2/n))
                    print("t_p ? t_T")
                    print(t_p)
                    print(self.t_table_student)
                    if t_p > self.t_table_student:
                        F += 1
                    else:
                        T += 1
                
        return T, F

# --- ЧАСТИНА 2: ГРАФІЧНИЙ ІНТЕРФЕЙС (GUI) ---
class KeyboardAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Вікно режиму перевірки клавіатурного почерку")
        self.root.geometry("650x300")
        
        self.logic = KeyboardAuthLogic()
        
        self.target_phrase = "тест"
        self.input_delays = []
        self.last_key_time = 0
        
        self.learning_attempts = []
        self.max_learning_attempts = 4
        
        self.auth_attempts = []
        self.max_auth_attempts = 3

        self.create_main_menu()
    
    def create_main_menu(self):
        self.clear_window()
        self.root.configure(bg="SystemButtonFace")
        
        tk.Label(self.root, text="Головне Меню", font=("Courier New", 16, "bold")).pack(pady=20)
        tk.Button(self.root, text="Режим навчання", width=20, height=2, command=self.open_study_mode).pack(pady=10)
        tk.Button(self.root, text="Режим перевірки", width=20, height=2, command=self.open_auth_mode).pack(pady=10)
        tk.Button(self.root, text="Вихід", width=20, height=2, command=self.root.quit).pack(pady=10)

    # --- РЕЖИМ НАВЧАННЯ ---
    def open_study_mode(self):
        self.clear_window()
        self.learning_attempts = []

        tk.Label(self.root, text="РЕЖИМ НАВЧАННЯ", font=("Courier New", 16, "bold")).place(x=150, y=10)
        
        tk.Label(self.root, text="Кодове слово:", font=("Courier New", 14, "bold")).place(x=20, y=50)
        tk.Label(self.root, text=self.target_phrase, font=("Courier New", 14, "bold"), fg="#CA1D37").place(x=180, y=50)

        tk.Label(self.root, text="Кількість введених симв:", font=("Courier New", 14, "bold")).place(x=20, y=90)
        self.lbl_symbol_count = tk.Label(self.root, text="0", font=("Courier New", 14, "bold"))
        self.lbl_symbol_count.place(x=320, y=90)

        self.entry = tk.Entry(self.root, font=("Courier New", 12), bg="#E6F0FA", width=25)
        self.entry.place(x=20, y=170, height=30)
        self.entry.bind('<KeyPress>', self.on_key_press)
        self.entry.bind('<KeyRelease>', self.update_char_counter)
        self.entry.focus_set()

        tk.Label(self.root, text="Кількість спроб:", font=("Courier New", 14, "bold")).place(x=20, y=220)
        self.combo_attempts = ttk.Combobox(self.root, values=["3", "10", "20"], width=5, state="readonly", font=("Courier New", 12))
        self.combo_attempts.current(0)
        self.combo_attempts.place(x=220, y=220)
        self.combo_attempts.bind("<<ComboboxSelected>>", self.update_max_attempts)
        self.max_learning_attempts = int(self.combo_attempts.get())

        btn_exit = tk.Button(self.root, text="Вийти з режиму", font=("Courier New", 12, "bold"), command=self.create_main_menu)
        btn_exit.place(x=20, y=260, width=200, height=35) # Трохи посунув вниз

        self.lbl_counter = tk.Label(self.root, text=f"Прогрес: 0 / {self.max_learning_attempts}", font=("Courier New", 10))
        self.lbl_counter.place(x=240, y=270)

    # --- РЕЖИМ ПЕРЕВІРКИ ---
    def open_auth_mode(self):
        self.clear_window()

        # Стиль шрифтів
        font_text = ("Courier New", 12, "bold")
        font_val = ("Courier New", 12)
        # --- ЗАГОЛОВОК ---
        tk.Label(self.root, text="РЕЖИМ ПЕРЕВІРКИ", font=("Courier New", 16, "bold")).pack(pady=10)

        # --- ЛІВА КОЛОНКА ---
        # Кодове слово
        tk.Label(self.root, text="Кодове слово:", font=font_text).place(x=20, y=50)
        tk.Label(self.root, text=self.target_phrase, font=font_text, fg="red").place(x=170, y=50)

        # К-сть спроб та Alpha
        tk.Label(self.root, text="К-сть спроб:", font=font_text).place(x=20, y=85)
        self.combo_attempts_check = ttk.Combobox(self.root, values=["1", "3", "5"], width=3, state="readonly", font=font_val)
        self.combo_attempts_check.set("3")
        self.combo_attempts_check.bind("<<ComboboxSelected>>", self.update_max_auth_attempts)
        self.max_auth_attempts = int(self.combo_attempts_check.get())
        self.combo_attempts_check.place(x=160, y=85)

        tk.Label(self.root, text="Alpha:", font=font_text).place(x=220, y=85)
        self.alpha_combo = ttk.Combobox(self.root, values=["0.20", "0.10", "0.05", "0.01", "0.005", "0.0001"], width=5, state="readonly", font=font_val)
        self.alpha_combo.set("0.05")
        self.alpha_combo.place(x=290, y=85)

        # Кількість введених символів
        tk.Label(self.root, text="Кількість введених символів:", font=font_text).place(x=20, y=120)
        self.lbl_auth_count = tk.Label(self.root, text="0", font=font_text)
        self.lbl_auth_count.place(x=300, y=120)

        # Поле вводу 
        self.entry = tk.Entry(self.root, font=("Courier New", 12), bg="#D3D8D3", width=30)
        self.entry.place(x=20, y=150, height=30)
        self.entry.bind('<KeyPress>', self.on_key_press)
        self.entry.bind('<KeyRelease>', self.update_auth_char_counter)
        self.entry.focus_set()

        # reset
        self.auth_attempts = []

        # Кнопка виходу
        btn_exit = tk.Button(self.root, text="Вийти з режиму для перевірки", font=("Courier New", 11, "bold"), command=self.create_main_menu)
        btn_exit.place(x=20, y=200, width=280, height=40)


        tk.Label(self.root, text="Статистичний аналіз", font=font_text).place(x=400, y=50)

        self.lbl_stat_p = tk.Label(self.root, text="P ідентифікації: -", font=font_text)
        self.lbl_stat_p.place(x=380, y=90)

        self.lbl_stat_err1 = tk.Label(self.root, text="Помилка 1-го роду: -", font=font_text)
        self.lbl_stat_err1.place(x=380, y=130)

        self.lbl_stat_err2 = tk.Label(self.root, text="Помилка 2-го роду: -", font=font_text)
        self.lbl_stat_err2.place(x=380, y=165)


    def update_char_counter(self, event=None):
        if hasattr(self, 'lbl_symbol_count') and self.lbl_symbol_count.winfo_exists():
            self.lbl_symbol_count.config(text=str(len(self.entry.get())))

    def update_auth_char_counter(self, event=None):
        if hasattr(self, 'lbl_auth_count') and self.lbl_auth_count.winfo_exists():
            self.lbl_auth_count.config(text=str(len(self.entry.get())))

    def update_max_attempts(self, event=None):
        self.max_learning_attempts = int(self.combo_attempts.get())
        count = len(self.learning_attempts)
        self.lbl_counter.config(text=f"Прогрес: {count} / {self.max_learning_attempts}")
    
    def update_max_auth_attempts(self, event=None):
        self.max_auth_attempts = int(self.combo_attempts_check.get())

    def on_key_press(self, event):
        current_time = time.time()
        
        if event.keysym == 'Return':
            text = self.entry.get()
            
            if text != self.target_phrase:
                messagebox.showerror("Помилка", f"Фраза введена невірно!\nПотрібно: {self.target_phrase}")
                self.reset_input()
                return

            # Обробка даних (пропускаємо перший інтервал)
            data_to_process = self.input_delays[1:] if len(self.input_delays) > 0 else []
            
            # Передаємо керування логіці обробки
            self.process_attempt(data_to_process)
            
            if self.entry.winfo_exists():
                self.reset_input()
                # Оновити лічильники в 0 після очищення
                if hasattr(self, 'lbl_auth_count') and self.lbl_auth_count.winfo_exists():
                    self.lbl_auth_count.config(text="0")
                if hasattr(self, 'lbl_symbol_count') and self.lbl_symbol_count.winfo_exists():
                    self.lbl_symbol_count.config(text="0")
            return

        if len(event.char) == 1:
            if self.last_key_time == 0:
                delay = 0.0
            else:
                delay = current_time - self.last_key_time
            
            self.input_delays.append(delay)
            self.last_key_time = current_time

    def reset_input(self):
        if self.entry.winfo_exists():
            self.entry.delete(0, tk.END)
        self.input_delays = []
        self.last_key_time = 0

    def process_attempt(self, delays):
        # 1. Режим Навчання
        if hasattr(self, 'lbl_counter') and self.lbl_counter.winfo_exists():
            if len(delays) != len(self.target_phrase)-1:
                messagebox.showwarning("Увага", "Кількість натискань не співпадає.")
                return

            self.learning_attempts.append(delays)
            count = len(self.learning_attempts)
            self.lbl_counter.config(text=f"Прогрес: {count} / {self.max_learning_attempts}")
            
            if count >= self.max_learning_attempts:
                self.logic.save_template(self.learning_attempts)
                messagebox.showinfo("Успіх", "Навчання завершено! Еталон збережено.")
                self.create_main_menu()

        # 2. Режим Перевірки
        elif hasattr(self, 'lbl_stat_p') and self.lbl_stat_p.winfo_exists():
            if len(delays) != len(self.target_phrase)-1:
                messagebox.showwarning("Увага", "Кількість натискань не співпадає.")
                return
            self.auth_attempts.append(delays)
            count = len(self.auth_attempts)
            
            print(self.auth_attempts)
            print("="*36)
            print(count)
            
            if count >= self.max_auth_attempts:
                template = self.logic.load_template()
                print(template)
                
                # Отримуємо результати перевірки
                alpha_val = np.float32(self.alpha_combo.get())
                
                self.logic.f_table_fisher = stats.f.ppf(1 - alpha_val/2, 
                                                        len(self.target_phrase) - 2, 
                                                        len(self.target_phrase) - 2)
                self.logic.t_table_student = stats.t.ppf(1 - alpha_val/2,  
                                                         len(self.target_phrase) - 2)
                
                T, F = self.logic.verify_user(self.auth_attempts, template)
            
                # Оновлення полів статистики

                # P ідентифікації
                self.lbl_stat_p.config(text=f"T/F: {T/(T+F):.2f}")

                alpha_val = self.alpha_combo.get()
                self.lbl_stat_err1.config(text=f"Помилка 1-го роду: {T/(T+F):.3f}")

                self.lbl_stat_err2.config(text=f"Помилка 2-го роду: {F/(T+F):.3f}")

                if T>F:
                    self.lbl_stat_p.config(fg="green")
                else:
                    self.lbl_stat_p.config(fg="red")

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.input_delays = []
        self.last_key_time = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = KeyboardAuthApp(root)
    root.mainloop()