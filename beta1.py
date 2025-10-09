import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import os


class EEGAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор ЭЭГ - СПМ спектральный анализ + Просмотр ЭЭГ")
        self.root.geometry("1600x1200")  # Увеличил размер окна

        self.data = None
        self.channel_names = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
        self.current_view = 'analysis'  # 'analysis' или 'eeg'
        self.current_channel = 0  # Текущий выбранный канал
        self.fs = 5000  # Частота дискретизации
        self.eeg_display_seconds = 10  # Секунд для отображения ЭЭГ
        self.eeg_start_time = 0  # Начальное время для отображения ЭЭГ

        # Обновленные диапазоны частот согласно стандартам
        self.freq_bands = {
            'delta': (0.5, 3.0),
            'theta': (4.0, 7.0),
            'alpha': (8.0, 13.0),
            'beta': (15.0, 30.0),
            'gamma': (30.0, 45.0)
        }

        self.create_widgets()

    def create_widgets(self):
        # Основной контейнер с разделением на две части
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая панель - графики и управление
        left_frame = tk.Frame(main_paned)
        main_paned.add(left_frame, width=1000)  # Фиксированная ширина для графиков

        # Правая панель - результаты (увеличим ширину)
        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, width=600)  # Увеличил ширину панели результатов

        # ЗАГОЛОВОК
        title_label = tk.Label(left_frame, text="Анализатор ЭЭГ данных - Спектральный анализ + Детальный просмотр ЭЭГ",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=15)

        # Фрейм для кнопок в две строки
        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)

        # Первая строка кнопок
        button_row1 = tk.Frame(button_frame)
        button_row1.pack(pady=5)

        # Кнопка загрузки файла
        load_btn = tk.Button(button_row1, text="ЗАГРУЗИТЬ ФАЙЛ .ASC",
                             command=self.load_file,
                             font=("Arial", 11, "bold"),
                             width=18,
                             height=2,
                             bg="#4CAF50",
                             fg="white",
                             activebackground="#45a049")
        load_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка просмотра исходной ЭЭГ
        self.view_eeg_btn = tk.Button(button_row1, text="ПРОСМОТР ЭЭГ",
                                      command=self.view_raw_eeg,
                                      font=("Arial", 11, "bold"),
                                      width=18,
                                      height=2,
                                      bg="#607D8B",
                                      fg="white",
                                      activebackground="#546E7A",
                                      state="disabled")
        self.view_eeg_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка анализа дельта-ритма
        self.analyze_delta_btn = tk.Button(button_row1, text="ДЕЛЬТА-РИТМ",
                                           command=lambda: self.analyze_data('delta'),
                                           font=("Arial", 11, "bold"),
                                           width=18,
                                           height=2,
                                           bg="#2196F3",
                                           fg="white",
                                           activebackground="#0b7dda",
                                           state="disabled")
        self.analyze_delta_btn.pack(side=tk.LEFT, padx=5)

        # Вторая строка кнопок
        button_row2 = tk.Frame(button_frame)
        button_row2.pack(pady=5)

        # Кнопка полного спектра
        self.analyze_full_btn = tk.Button(button_row2, text="ПОЛНЫЙ СПЕКТР",
                                          command=lambda: self.analyze_data('full_spectrum'),
                                          font=("Arial", 11, "bold"),
                                          width=18,
                                          height=2,
                                          bg="#9C27B0",
                                          fg="white",
                                          activebackground="#7B1FA2",
                                          state="disabled")
        self.analyze_full_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка сохранения результатов
        self.save_btn = tk.Button(button_row2, text="СОХРАНИТЬ РЕЗУЛЬТАТЫ",
                                  command=self.save_results,
                                  font=("Arial", 11, "bold"),
                                  width=18,
                                  height=2,
                                  bg="#FF9800",
                                  fg="white",
                                  state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка возврата к анализу
        self.back_btn = tk.Button(button_row2, text="ВЕРНУТЬСЯ К АНАЛИЗУ",
                                  command=self.show_analysis_view,
                                  font=("Arial", 11, "bold"),
                                  width=18,
                                  height=2,
                                  bg="#795548",
                                  fg="white",
                                  state="disabled")
        self.back_btn.pack(side=tk.LEFT, padx=5)

        # Фрейм управления просмотром ЭЭГ
        self.eeg_control_frame = tk.Frame(button_frame)

        # Кнопки переключения каналов
        channel_frame = tk.Frame(self.eeg_control_frame)
        channel_frame.pack(pady=5)

        tk.Label(channel_frame, text="Каналы:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.channel_buttons = []
        for i, channel in enumerate(self.channel_names):
            btn = tk.Button(channel_frame, text=channel,
                            command=lambda idx=i: self.switch_channel(idx),
                            font=("Arial", 9, "bold"),
                            width=6,
                            height=1,
                            bg="#E0E0E0",
                            fg="black",
                            state="disabled")
            btn.pack(side=tk.LEFT, padx=2)
            self.channel_buttons.append(btn)

        # Управление временным окном
        time_frame = tk.Frame(self.eeg_control_frame)
        time_frame.pack(pady=5)

        tk.Label(time_frame, text="Окно просмотра:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.time_buttons = []
        time_options = [5, 10, 30, 60]
        for seconds in time_options:
            btn = tk.Button(time_frame, text=f"{seconds} сек",
                            command=lambda s=seconds: self.set_time_window(s),
                            font=("Arial", 9),
                            width=8,
                            height=1,
                            bg="#F5F5F5",
                            fg="black",
                            state="disabled")
            btn.pack(side=tk.LEFT, padx=2)
            self.time_buttons.append(btn)

        # Управление скроллингом
        scroll_frame = tk.Frame(self.eeg_control_frame)
        scroll_frame.pack(pady=5)

        self.scroll_left_btn = tk.Button(scroll_frame, text="◀ НАЗАД",
                                         command=self.scroll_left,
                                         font=("Arial", 9),
                                         width=10,
                                         height=1,
                                         bg="#BBDEFB",
                                         fg="black",
                                         state="disabled")
        self.scroll_left_btn.pack(side=tk.LEFT, padx=2)

        self.scroll_right_btn = tk.Button(scroll_frame, text="ВПЕРЕД ▶",
                                          command=self.scroll_right,
                                          font=("Arial", 9),
                                          width=10,
                                          height=1,
                                          bg="#BBDEFB",
                                          fg="black",
                                          state="disabled")
        self.scroll_right_btn.pack(side=tk.LEFT, padx=2)

        self.zoom_out_btn = tk.Button(scroll_frame, text="− УВЕЛИЧИТЬ ОКНО",
                                      command=self.zoom_out,
                                      font=("Arial", 9),
                                      width=15,
                                      height=1,
                                      bg="#C8E6C9",
                                      fg="black",
                                      state="disabled")
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)

        self.zoom_in_btn = tk.Button(scroll_frame, text="+ УМЕНЬШИТЬ ОКНО",
                                     command=self.zoom_in,
                                     font=("Arial", 9),
                                     width=15,
                                     height=1,
                                     bg="#FFCDD2",
                                     fg="black",
                                     state="disabled")
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)

        # Информация о файле
        self.file_label = tk.Label(left_frame, text="Файл не загружен",
                                   font=("Arial", 12))
        self.file_label.pack(pady=5)

        # Область для графиков в левой панели
        self.create_plot_area(left_frame)

        # Область для результатов в правой панели (УВЕЛИЧЕНА)
        self.create_results_area(right_frame)

    def create_plot_area(self, parent):
        # Фрейм для графиков
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Создаем фигуру matplotlib
        self.fig = plt.figure(figsize=(12, 9))  # Немного уменьшил для баланса

        # Изначально создаем сетку для анализа
        self.setup_analysis_grid()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_analysis_grid(self):
        """Настраивает сетку для анализа спектра"""
        self.fig.clear()
        self.gs = plt.GridSpec(3, 3, figure=self.fig)

        # Основные оси для графиков анализа
        self.ax_hist = self.fig.add_subplot(self.gs[0, :])
        self.ax_psd = []
        for i in range(6):
            row = 1 + i // 3
            col = i % 3
            self.ax_psd.append(self.fig.add_subplot(self.gs[row, col]))

        self.fig.tight_layout(pad=3.0)

    def setup_eeg_grid(self):
        """Настраивает сетку для просмотра ЭЭГ"""
        self.fig.clear()
        # Основной график ЭЭГ
        self.ax_eeg = self.fig.add_subplot(111)

        # Мини-карта всего сигнала
        self.ax_minimap = self.fig.add_axes([0.1, 0.92, 0.8, 0.06])  # [left, bottom, width, height]

        self.fig.tight_layout(pad=3.0)

    def create_results_area(self, parent):
        # Фрейм для результатов
        results_frame = tk.Frame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Заголовок области результатов
        self.results_label = tk.Label(results_frame, text="РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:",
                                      font=("Arial", 14, "bold"))
        self.results_label.pack(anchor='w', pady=(0, 10))

        # Текстовое поле для результатов (УВЕЛИЧЕНО)
        self.results_text = tk.Text(results_frame,
                                    height=35,  # Увеличил высоту
                                    width=80,  # Увеличил ширину
                                    font=("Courier", 12),  # Увеличил шрифт
                                    wrap=tk.WORD,
                                    bg="#f8f9fa",
                                    relief=tk.SUNKEN,
                                    bd=2,
                                    padx=10,  # Отступы внутри текстового поля
                                    pady=10)
        self.results_text.pack(fill='both', expand=True)

        # Фрейм для кнопок под текстовым полем
        results_buttons_frame = tk.Frame(results_frame)
        results_buttons_frame.pack(fill=tk.X, pady=(10, 0))

        # Кнопка очистки результатов
        clear_btn = tk.Button(results_buttons_frame,
                              text="ОЧИСТИТЬ РЕЗУЛЬТАТЫ",
                              command=self.clear_results,
                              font=("Arial", 10, "bold"),
                              width=20,
                              height=1,
                              bg="#F44336",
                              fg="white")
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка копирования результатов
        copy_btn = tk.Button(results_buttons_frame,
                             text="КОПИРОВАТЬ РЕЗУЛЬТАТЫ",
                             command=self.copy_results,
                             font=("Arial", 10, "bold"),
                             width=20,
                             height=1,
                             bg="#2196F3",
                             fg="white")
        copy_btn.pack(side=tk.LEFT, padx=5)

        # Прокрутка для текстового поля
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        # Горизонтальная прокрутка
        h_scrollbar = tk.Scrollbar(self.results_text, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_text.config(xscrollcommand=h_scrollbar.set)
        h_scrollbar.config(command=self.results_text.xview)

    def clear_results(self):
        """Очищает поле результатов"""
        self.results_text.delete(1.0, tk.END)

    def copy_results(self):
        """Копирует результаты в буфер обмена"""
        results = self.results_text.get(1.0, tk.END)
        if results.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(results)
            messagebox.showinfo("Успех", "Результаты скопированы в буфер обмена!")
        else:
            messagebox.showwarning("Внимание", "Нет данных для копирования")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл .asc",
            filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Чтение файла с обработкой заголовка
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                # Пропускаем строки заголовка и читаем данные
                data_lines = []
                for line in lines:
                    if not line.startswith(';') and line.strip():
                        data_lines.append(line.strip())

                # Преобразуем данные в числовой формат
                data = []
                for line in data_lines:
                    if line.strip():
                        row = [float(x) for x in line.split()]
                        data.append(row)

                self.data = np.array(data)
                self.total_duration = len(self.data) / self.fs

                self.file_label.config(text=f"Загружен: {os.path.basename(file_path)}")
                self.analyze_delta_btn.config(state="normal")
                self.analyze_full_btn.config(state="normal")
                self.save_btn.config(state="normal")
                self.view_eeg_btn.config(state="normal")
                self.back_btn.config(state="normal")

                # Активируем кнопки управления ЭЭГ
                for btn in self.channel_buttons:
                    btn.config(state="normal", bg="#E0E0E0")
                for btn in self.time_buttons:
                    btn.config(state="normal")
                self.scroll_left_btn.config(state="normal")
                self.scroll_right_btn.config(state="normal")
                self.zoom_in_btn.config(state="normal")
                self.zoom_out_btn.config(state="normal")

                # Активируем первый канал
                self.channel_buttons[0].config(bg="#2196F3", fg="white")

                messagebox.showinfo("Успех",
                                    f"Файл загружен!\nКаналов: {self.data.shape[1]}\nОтсчетов: {self.data.shape[0]}\nДлительность: {self.total_duration:.2f} сек")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def switch_channel(self, channel_idx):
        """Переключает отображаемый канал ЭЭГ"""
        if self.data is None or self.current_view != 'eeg':
            return

        self.current_channel = channel_idx

        # Обновляем цвета кнопок
        for i, btn in enumerate(self.channel_buttons):
            if i == channel_idx:
                btn.config(bg="#2196F3", fg="white")
            else:
                btn.config(bg="#E0E0E0", fg="black")

        # Перерисовываем ЭЭГ
        self.update_eeg_display()

    def set_time_window(self, seconds):
        """Устанавливает временное окно для отображения"""
        if self.data is None or self.current_view != 'eeg':
            return

        self.eeg_display_seconds = seconds
        self.update_eeg_display()

    def scroll_left(self):
        """Прокрутка назад во времени"""
        if self.data is None or self.current_view != 'eeg':
            return

        new_start = max(0, self.eeg_start_time - self.eeg_display_seconds * 0.5)
        if new_start != self.eeg_start_time:
            self.eeg_start_time = new_start
            self.update_eeg_display()

    def scroll_right(self):
        """Прокрутка вперед во времени"""
        if self.data is None or self.current_view != 'eeg':
            return

        max_start = self.total_duration - self.eeg_display_seconds
        new_start = min(max_start, self.eeg_start_time + self.eeg_display_seconds * 0.5)
        if new_start != self.eeg_start_time:
            self.eeg_start_time = new_start
            self.update_eeg_display()

    def zoom_in(self):
        """Уменьшает временное окно (зум внутрь)"""
        if self.data is None or self.current_view != 'eeg':
            return

        if self.eeg_display_seconds > 1:
            self.eeg_display_seconds = max(1, self.eeg_display_seconds // 2)
            self.update_eeg_display()

    def zoom_out(self):
        """Увеличивает временное окно (зум наружу)"""
        if self.data is None or self.current_view != 'eeg':
            return

        if self.eeg_display_seconds < 300:  # Максимум 5 минут
            self.eeg_display_seconds = min(300, self.eeg_display_seconds * 2)
            self.update_eeg_display()

    def view_raw_eeg(self):
        """Просмотр исходных сигналов ЭЭГ"""
        if self.data is None:
            return

        try:
            self.current_view = 'eeg'
            self.eeg_start_time = 0  # Сбрасываем к началу

            # Настраиваем сетку для ЭЭГ
            self.setup_eeg_grid()

            # Показываем управление ЭЭГ
            self.eeg_control_frame.pack(pady=10)

            # Обновляем отображение
            self.update_eeg_display()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось отобразить ЭЭГ:\n{str(e)}")

    def update_eeg_display(self):
        """Обновляет отображение ЭЭГ"""
        if self.data is None or self.current_view != 'eeg':
            return

        try:
            # Очищаем графики
            self.ax_eeg.clear()
            self.ax_minimap.clear()

            # Получаем данные текущего канала
            channel_data = self.data[:, self.current_channel]
            time = np.arange(len(channel_data)) / self.fs

            # Определяем диапазон для отображения
            start_idx = int(self.eeg_start_time * self.fs)
            end_idx = int((self.eeg_start_time + self.eeg_display_seconds) * self.fs)
            end_idx = min(end_idx, len(channel_data))

            # Данные для основного графика
            display_data = channel_data[start_idx:end_idx]
            display_time = time[start_idx:end_idx]

            # Основной график ЭЭГ
            self.ax_eeg.plot(display_time, display_data, color='#2196F3', linewidth=1)
            self.ax_eeg.set_title(f'ЭЭГ - Канал {self.channel_names[self.current_channel]}',
                                  fontweight='bold', fontsize=14)
            self.ax_eeg.set_ylabel('Амплитуда (мкВ)', fontsize=12)
            self.ax_eeg.set_xlabel('Время (секунды)', fontsize=12)
            self.ax_eeg.grid(True, alpha=0.3)

            # Автоматическое масштабирование по Y
            y_margin = (np.max(display_data) - np.min(display_data)) * 0.1
            if y_margin == 0:  # Если сигнал постоянный
                y_margin = 1
            self.ax_eeg.set_ylim(np.min(display_data) - y_margin, np.max(display_data) + y_margin)

            # Мини-карта всего сигнала
            self.ax_minimap.plot(time, channel_data, color='gray', linewidth=0.5, alpha=0.7)

            # Показываем текущее окно на мини-карте
            window_start = self.eeg_start_time
            window_end = self.eeg_start_time + self.eeg_display_seconds

            self.ax_minimap.axvspan(window_start, window_end, alpha=0.3, color='red')
            self.ax_minimap.set_xlim(0, self.total_duration)
            self.ax_minimap.set_ylabel('Вся запись', fontsize=8)
            self.ax_minimap.tick_params(axis='both', which='major', labelsize=6)
            self.ax_minimap.grid(True, alpha=0.2)

            # Убираем метки по Y на мини-карте для экономии места
            self.ax_minimap.set_yticklabels([])

            self.fig.tight_layout()
            self.canvas.draw()

            # Обновляем информацию
            self.update_eeg_info()

        except Exception as e:
            print(f"Ошибка при обновлении ЭЭГ: {e}")

    def update_eeg_info(self):
        """Обновляет информацию о ЭЭГ в текстовом поле"""
        self.results_text.delete(1.0, tk.END)

        channel_data = self.data[:, self.current_channel]
        current_segment = channel_data[int(self.eeg_start_time * self.fs):int(
            (self.eeg_start_time + self.eeg_display_seconds) * self.fs)]

        info_text = f"ПРОСМОТР ЭЭГ - КАНАЛ {self.channel_names[self.current_channel]}\n"
        info_text += "=" * 80 + "\n\n"
        info_text += f"ОБЩАЯ ИНФОРМАЦИЯ:\n"
        info_text += f"• Канал: {self.channel_names[self.current_channel]}\n"
        info_text += f"• Длительность записи: {self.total_duration:.2f} сек\n"
        info_text += f"• Текущее окно: {self.eeg_start_time:.1f}-{self.eeg_start_time + self.eeg_display_seconds:.1f} сек\n"
        info_text += f"• Размер окна: {self.eeg_display_seconds} сек\n"
        info_text += f"• Частота дискретизации: {self.fs} Гц\n"
        info_text += f"• Всего отсчетов: {len(self.data)}\n"

        info_text += f"\nСТАТИСТИКА ТЕКУЩЕГО ОКНА:\n"
        info_text += "-" * 50 + "\n"
        info_text += f"• Минимум: {np.min(current_segment):.2f} мкВ\n"
        info_text += f"• Максимум: {np.max(current_segment):.2f} мкВ\n"
        info_text += f"• Среднее: {np.mean(current_segment):.2f} мкВ\n"
        info_text += f"• Стандартное отклонение: {np.std(current_segment):.2f} мкВ\n"
        info_text += f"• Динамический диапазон: {np.max(current_segment) - np.min(current_segment):.2f} мкВ\n"

        info_text += f"\nСТАТИСТИКА ВСЕГО СИГНАЛА:\n"
        info_text += "-" * 45 + "\n"
        info_text += f"• Минимум: {np.min(channel_data):.2f} мкВ\n"
        info_text += f"• Максимум: {np.max(channel_data):.2f} мкВ\n"
        info_text += f"• Среднее: {np.mean(channel_data):.2f} мкВ\n"
        info_text += f"• Стандартное отклонение: {np.std(channel_data):.2f} мкВ\n"

        info_text += f"\nУПРАВЛЕНИЕ ПРОСМОТРОМ:\n"
        info_text += "-" * 40 + "\n"
        info_text += "• Кнопки каналов - переключение между электродами\n"
        info_text += "• 5/10/30/60 сек - размер временного окна\n"
        info_text += "◀ НАЗАД/ВПЕРЕД ▶ - прокрутка по времени\n"
        info_text += "+/- - изменение размера окна (зум)\n"
        info_text += "• Красная область на мини-карте - текущее положение\n"
        info_text += "• ОЧИСТИТЬ РЕЗУЛЬТАТЫ - очистка этого поля\n"
        info_text += "• КОПИРОВАТЬ РЕЗУЛЬТАТЫ - копирование в буфер обмена\n"

        self.results_text.insert(1.0, info_text)
        self.results_label.config(text=f"ПРОСМОТР ЭЭГ - {self.channel_names[self.current_channel]}:")

    def show_analysis_view(self):
        """Возврат к виду анализа"""
        if self.data is None:
            return

        self.current_view = 'analysis'

        # Восстанавливаем исходную конфигурацию графиков
        self.setup_analysis_grid()
        self.results_label.config(text="РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Готов к анализу. Выберите тип анализа выше.")

        # Скрываем управление ЭЭГ
        self.eeg_control_frame.pack_forget()

        self.canvas.draw()

    # Остальные методы остаются без изменений
    def analyze_data(self, analysis_type):
        if self.data is None:
            return

        # Переключаемся на вид анализа
        self.show_analysis_view()

        self.results_text.delete(1.0, tk.END)
        self.current_view = 'analysis'

        try:
            fs = self.fs
            powers = []
            all_psd_data = []
            all_freqs_data = []

            if analysis_type == 'delta':
                results_text = "РЕЗУЛЬТАТЫ АНАЛИЗА СПМ ДЕЛЬТА-РИТМА (0.5-3 Гц)\n"
                freq_range = self.freq_bands['delta']
                self.results_label.config(text="РЕЗУЛЬТАТЫ АНАЛИЗА ДЕЛЬТА-РИТМА:")
            else:
                results_text = "РЕЗУЛЬТАТЫ АНАЛИЗА ПОЛНОГО СПЕКТРА ЭЭГ (0.5-45 Гц)\n"
                freq_range = (0.5, 45.0)
                self.results_label.config(text="РЕЗУЛЬТАТЫ АНАЛИЗА ПОЛНОГО СПЕКТРА:")

            results_text += "=" * 80 + "\n"
            results_text += "ДИАПАЗОНЫ ЧАСТОТ ДЛЯ АНАЛИЗА:\n"
            results_text += f"  Дельта (Δ): {self.freq_bands['delta'][0]}-{self.freq_bands['delta'][1]} Гц\n"
            results_text += f"  Тета (θ): {self.freq_bands['theta'][0]}-{self.freq_bands['theta'][1]} Гц\n"
            results_text += f"  Альфа (α): {self.freq_bands['alpha'][0]}-{self.freq_bands['alpha'][1]} Гц\n"
            results_text += f"  Бета (β): {self.freq_bands['beta'][0]}-{self.freq_bands['beta'][1]} Гц\n"
            results_text += f"  Гамма (γ): {self.freq_bands['gamma'][0]}-{self.freq_bands['gamma'][1]} Гц\n"
            results_text += "=" * 80 + "\n\n"

            # Анализируем каждый канал
            for i, name in enumerate(self.channel_names):
                signal = self.data[:, i]

                # Используем весь сигнал
                if len(signal) > 0:
                    # Убираем постоянную составляющую (DC offset)
                    signal = signal - np.mean(signal)

                    N = len(signal)

                    # Применяем окно Ханна для уменьшения утечки спектра
                    window = np.hanning(N)
                    signal_windowed = signal * window

                    # Вычисление СПМ с нормализацией
                    fft_result = np.fft.fft(signal_windowed)

                    # Расчет СПМ
                    psd = (np.real(fft_result) ** 2 + np.imag(fft_result) ** 2)
                    psd = psd / (fs * np.sum(window ** 2))

                    frequencies = np.fft.fftfreq(N, 1 / fs)

                    # Сохраняем данные для построения графиков
                    positive_idx = frequencies >= 0
                    freqs_positive = frequencies[positive_idx]
                    psd_positive = psd[positive_idx]

                    all_psd_data.append(psd_positive)
                    all_freqs_data.append(freqs_positive)

                    # Мощность в выбранном диапазоне
                    freq_mask = (freqs_positive >= freq_range[0]) & (freqs_positive <= freq_range[1])
                    power = np.trapezoid(psd_positive[freq_mask], freqs_positive[freq_mask])

                    powers.append(power)

                    if analysis_type == 'delta':
                        results_text += f"🔹 {name}: {power:.2f} мкВ²/Гц\n"
                    else:
                        # Для полного спектра показываем мощность всех ритмов
                        delta_power = np.trapezoid(
                            psd_positive[(freqs_positive >= self.freq_bands['delta'][0]) &
                                         (freqs_positive <= self.freq_bands['delta'][1])],
                            freqs_positive[(freqs_positive >= self.freq_bands['delta'][0]) &
                                           (freqs_positive <= self.freq_bands['delta'][1])]
                        )
                        theta_power = np.trapezoid(
                            psd_positive[(freqs_positive >= self.freq_bands['theta'][0]) &
                                         (freqs_positive <= self.freq_bands['theta'][1])],
                            freqs_positive[(freqs_positive >= self.freq_bands['theta'][0]) &
                                           (freqs_positive <= self.freq_bands['theta'][1])]
                        )
                        alpha_power = np.trapezoid(
                            psd_positive[(freqs_positive >= self.freq_bands['alpha'][0]) &
                                         (freqs_positive <= self.freq_bands['alpha'][1])],
                            freqs_positive[(freqs_positive >= self.freq_bands['alpha'][0]) &
                                           (freqs_positive <= self.freq_bands['alpha'][1])]
                        )
                        beta_power = np.trapezoid(
                            psd_positive[(freqs_positive >= self.freq_bands['beta'][0]) &
                                         (freqs_positive <= self.freq_bands['beta'][1])],
                            freqs_positive[(freqs_positive >= self.freq_bands['beta'][0]) &
                                           (freqs_positive <= self.freq_bands['beta'][1])]
                        )
                        gamma_power = np.trapezoid(
                            psd_positive[(freqs_positive >= self.freq_bands['gamma'][0]) &
                                         (freqs_positive <= self.freq_bands['gamma'][1])],
                            freqs_positive[(freqs_positive >= self.freq_bands['gamma'][0]) &
                                           (freqs_positive <= self.freq_bands['gamma'][1])]
                        )

                        results_text += f"🔹 {name}:\n"
                        results_text += f"   Δ: {delta_power:8.2f} мкВ²/Гц | θ: {theta_power:8.2f} мкВ²/Гц\n"
                        results_text += f"   α: {alpha_power:8.2f} мкВ²/Гц | β: {beta_power:8.2f} мкВ²/Гц\n"
                        results_text += f"   γ: {gamma_power:8.2f} мкВ²/Гц\n\n"

                else:
                    powers.append(0)
                    all_psd_data.append(None)
                    all_freqs_data.append(None)
                    results_text += f"🔹 {name}: Нет данных\n\n"

            # Добавляем итоговую информацию
            results_text += "\n" + "=" * 80 + "\n"
            if analysis_type == 'delta':
                results_text += f"МАКСИМАЛЬНАЯ МОЩНОСТЬ ДЕЛЬТА-РИТМА: {max(powers):.2f} мкВ²/Гц ({self.channel_names[np.argmax(powers)]})\n"
                results_text += f"МИНИМАЛЬНАЯ МОЩНОСТЬ ДЕЛЬТА-РИТМА: {min(powers):.2f} мкВ²/Гц ({self.channel_names[np.argmin(powers)]})\n"
                results_text += f"СРЕДНЯЯ МОЩНОСТЬ ДЕЛЬТА-РИТМА: {np.mean(powers):.2f} мкВ²/Гц\n"
            else:
                results_text += f"ОБЩАЯ МОЩНОСТЬ СПЕКТРА (0.5-45 Гц): {np.mean(powers):.2f} мкВ²/Гц\n"

            # Обновляем текстовое поле
            self.results_text.insert(1.0, results_text)

            # Строим графики
            self.update_plots(powers, all_psd_data, all_freqs_data, analysis_type)

        except Exception as e:
            messagebox.showerror("Ошибка анализа", str(e))

    def update_plots(self, powers, all_psd_data, all_freqs_data, analysis_type):
        # Очищаем все графики
        self.ax_hist.clear()
        for ax in self.ax_psd:
            ax.clear()

        # График 1: Гистограмма (верхний график)
        if analysis_type == 'delta':
            title = 'Мощность дельта-ритма по каналам (0.5-3 Гц)'
            ylabel = 'Мощность (мкВ²/Гц)'
        else:
            title = 'Общая мощность спектра по каналам (0.5-45 Гц)'
            ylabel = 'Мощность (мкВ²/Гц)'

        bars = self.ax_hist.bar(self.channel_names, powers,
                                color='skyblue', edgecolor='navy')
        self.ax_hist.set_title(title, fontweight='bold', fontsize=14)
        self.ax_hist.set_ylabel(ylabel, fontsize=12)
        self.ax_hist.grid(True, alpha=0.3)

        # Добавляем значения на столбцы гистограммы
        for bar, power in zip(bars, powers):
            self.ax_hist.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + max(powers) * 0.01,
                              f'{power:.2f}', ha='center', va='bottom', fontsize=11)

        # Графики 2-7: СПМ для каждого канала
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        for i, (ax, name, color) in enumerate(zip(self.ax_psd, self.channel_names, colors)):
            if all_psd_data[i] is not None and all_freqs_data[i] is not None:
                # Рисуем график СПМ
                ax.plot(all_freqs_data[i], all_psd_data[i], color=color, linewidth=1)

                if analysis_type == 'delta':
                    ax.set_xlim(0, 6)
                    # Подсвечиваем дельта-диапазон
                    delta_mask = (all_freqs_data[i] >= self.freq_bands['delta'][0]) & (
                            all_freqs_data[i] <= self.freq_bands['delta'][1])
                    ax.fill_between(all_freqs_data[i][delta_mask],
                                    all_psd_data[i][delta_mask],
                                    alpha=0.3, color=color)
                    ax.set_title(f'{name} - Δ: {powers[i]:.2f}', fontsize=12, fontweight='bold')

                    # Вертикальные линии для дельта-ритма
                    ax.axvline(x=self.freq_bands['delta'][0], color='gray', linestyle='--', alpha=0.7)
                    ax.axvline(x=self.freq_bands['delta'][1], color='gray', linestyle='--', alpha=0.7)
                else:
                    ax.set_xlim(0, 50)
                    # Подсвечиваем все основные ритмы с новыми диапазонами
                    delta_mask = (all_freqs_data[i] >= self.freq_bands['delta'][0]) & (
                            all_freqs_data[i] <= self.freq_bands['delta'][1])
                    theta_mask = (all_freqs_data[i] >= self.freq_bands['theta'][0]) & (
                            all_freqs_data[i] <= self.freq_bands['theta'][1])
                    alpha_mask = (all_freqs_data[i] >= self.freq_bands['alpha'][0]) & (
                            all_freqs_data[i] <= self.freq_bands['alpha'][1])
                    beta_mask = (all_freqs_data[i] >= self.freq_bands['beta'][0]) & (
                            all_freqs_data[i] <= self.freq_bands['beta'][1])
                    gamma_mask = (all_freqs_data[i] >= self.freq_bands['gamma'][0]) & (
                            all_freqs_data[i] <= self.freq_bands['gamma'][1])

                    ax.fill_between(all_freqs_data[i][delta_mask], all_psd_data[i][delta_mask], alpha=0.3, color='red',
                                    label='Δ')
                    ax.fill_between(all_freqs_data[i][theta_mask], all_psd_data[i][theta_mask], alpha=0.3, color='blue',
                                    label='θ')
                    ax.fill_between(all_freqs_data[i][alpha_mask], all_psd_data[i][alpha_mask], alpha=0.3,
                                    color='green', label='α')
                    ax.fill_between(all_freqs_data[i][beta_mask], all_psd_data[i][beta_mask], alpha=0.3, color='orange',
                                    label='β')
                    ax.fill_between(all_freqs_data[i][gamma_mask], all_psd_data[i][gamma_mask], alpha=0.3,
                                    color='purple', label='γ')

                    ax.set_title(f'{name} - Полный спектр', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=8)

                # Безопасное вычисление максимума для оси Y
                if analysis_type == 'delta':
                    mask = all_freqs_data[i] <= 6
                else:
                    mask = all_freqs_data[i] <= 50

                if len(all_psd_data[i][mask]) > 0:
                    y_max = max(all_psd_data[i][mask]) * 1.1
                    ax.set_ylim(0, y_max)

                ax.set_xlabel('Частота (Гц)', fontsize=10)
                ax.set_ylabel('СПМ (мкВ²/Гц)', fontsize=10)
                ax.grid(True, alpha=0.3)

            else:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{name} - Нет данных', fontsize=10)

        # Обновляем canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def save_results(self):
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))

                # Также сохраняем график как изображение
                plot_path = file_path.replace('.txt', '_plot.png')
                self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')

                messagebox.showinfo("Успех", f"Результаты сохранены!\nТекст: {file_path}\nГрафик: {plot_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{str(e)}")


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalyzerApp(root)
    root.mainloop()