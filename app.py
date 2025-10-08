import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class EEGAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор ЭЭГ - СПМ дельта-ритма")
        self.root.geometry("1300x1000")

        self.data = None
        self.channel_names = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        title_label = tk.Label(self.root, text="Анализатор ЭЭГ данных",
                               font=("Arial", 20, "bold"))
        title_label.pack(pady=15)

        # Фрейм для кнопок в одну линию
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопка загрузки файла
        load_btn = tk.Button(button_frame, text="ЗАГРУЗИТЬ ФАЙЛ .ASC",
                             command=self.load_file,
                             font=("Arial", 14, "bold"),
                             width=22,
                             height=2,
                             bg="#4CAF50",
                             fg="white",
                             activebackground="#45a049")
        load_btn.pack(side=tk.LEFT, padx=10)

        # Кнопка анализа
        self.analyze_btn = tk.Button(button_frame, text="АНАЛИЗИРОВАТЬ СПМ",
                                     command=self.analyze_data,
                                     font=("Arial", 14, "bold"),
                                     width=22,
                                     height=2,
                                     bg="#2196F3",
                                     fg="white",
                                     activebackground="#0b7dda",
                                     state="disabled")
        self.analyze_btn.pack(side=tk.LEFT, padx=10)

        # Кнопка сохранения результатов
        self.save_btn = tk.Button(button_frame, text="СОХРАНИТЬ РЕЗУЛЬТАТЫ",
                                  command=self.save_results,
                                  font=("Arial", 14, "bold"),
                                  width=22,
                                  height=2,
                                  bg="#FF9800",
                                  fg="white",
                                  state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=10)

        # Информация о файле
        self.file_label = tk.Label(self.root, text="Файл не загружен",
                                   font=("Arial", 12))
        self.file_label.pack(pady=5)

        # Область для графиков
        self.create_plot_area()

        # Область для результатов
        self.create_results_area()

    def create_plot_area(self):
        # Фрейм для графиков
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Создаем фигуру matplotlib с 7 subplots (1 гистограмма + 6 графиков СПМ)
        self.fig = plt.figure(figsize=(14, 9))

        # Сетка 3x3: гистограмма занимает 3 ячейки в первом ряду, под ней 6 графиков
        self.gs = plt.GridSpec(3, 3, figure=self.fig)

        # Гистограмма (занимает весь первый ряд)
        self.ax_hist = self.fig.add_subplot(self.gs[0, :])

        # Графики СПМ для каждого канала (второй и третий ряды)
        self.ax_psd = []
        for i in range(6):
            row = 1 + i // 3  # 1 или 2
            col = i % 3  # 0, 1 или 2
            self.ax_psd.append(self.fig.add_subplot(self.gs[row, col]))

        self.fig.tight_layout(pad=3.0)

        # Встраиваем график в Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_results_area(self):
        # Фрейм для результатов
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Заголовок области результатов
        results_label = tk.Label(results_frame, text="РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:",
                                 font=("Arial", 14, "bold"))
        results_label.pack(anchor='w', pady=(0, 5))

        # Текстовое поле для результатов - УВЕЛИЧЕНО
        self.results_text = tk.Text(results_frame, height=12, font=("Courier", 12),
                                    wrap=tk.WORD, bg="#f8f9fa", relief=tk.SUNKEN, bd=2)
        self.results_text.pack(fill='both', expand=True)

        # Прокрутка для текстового поля
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл .asc",
            filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.data = np.loadtxt(file_path, skiprows=4)
                self.file_label.config(text=f"Загружен: {os.path.basename(file_path)}")
                self.analyze_btn.config(state="normal")
                self.save_btn.config(state="normal")
                messagebox.showinfo("Успех", f"Файл загружен!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def analyze_data(self):
        if self.data is None:
            return

        self.results_text.delete(1.0, tk.END)

        try:
            fs = 5000
            delta_powers = []
            all_psd_data = []  # Будем хранить данные СПМ для всех каналов
            all_freqs_data = []  # Будем хранить частоты для всех каналов

            results_text = "РЕЗУЛЬТАТЫ АНАЛИЗА СПМ ДЕЛЬТА-РИТМА\n"
            results_text += "=" * 60 + "\n\n"

            # Анализируем каждый канал
            for i, name in enumerate(self.channel_names):
                signal = self.data[:, i]
                signal = signal[signal != 0]

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
                    psd = psd / (fs * np.sum(window ** 2))  # Правильная нормализация

                    frequencies = np.fft.fftfreq(N, 1 / fs)

                    # Сохраняем данные для построения графиков
                    positive_idx = frequencies >= 0
                    freqs_positive = frequencies[positive_idx]
                    psd_positive = psd[positive_idx]

                    all_psd_data.append(psd_positive)
                    all_freqs_data.append(freqs_positive)

                    # Мощность дельта-ритма
                    delta_mask = (freqs_positive >= 0.5) & (freqs_positive <= 4.0)
                    delta_power = np.trapezoid(psd_positive[delta_mask], freqs_positive[delta_mask])

                    delta_powers.append(delta_power)
                    results_text += f"🔹 {name}: {delta_power:.2f} мкВ²/Гц\n"
                else:
                    delta_powers.append(0)
                    all_psd_data.append(None)
                    all_freqs_data.append(None)
                    results_text += f"🔹 {name}: Нет данных\n"

            # Добавляем итоговую информацию
            results_text += "\n" + "=" * 60 + "\n"
            results_text += f"МАКСИМАЛЬНАЯ МОЩНОСТЬ: {max(delta_powers):.2f} мкВ²/Гц ({self.channel_names[np.argmax(delta_powers)]})\n"
            results_text += f"МИНИМАЛЬНАЯ МОЩНОСТЬ: {min(delta_powers):.2f} мкВ²/Гц ({self.channel_names[np.argmin(delta_powers)]})\n"
            results_text += f"СРЕДНЯЯ МОЩНОСТЬ: {np.mean(delta_powers):.2f} мкВ²/Гц\n"

            # Обновляем текстовое поле
            self.results_text.insert(1.0, results_text)

            # Строим графики
            self.update_plots(delta_powers, all_psd_data, all_freqs_data)

        except Exception as e:
            messagebox.showerror("Ошибка анализа", str(e))

    def update_plots(self, delta_powers, all_psd_data, all_freqs_data):
        # Очищаем все графики
        self.ax_hist.clear()
        for ax in self.ax_psd:
            ax.clear()

        # График 1: Гистограмма (верхний график)
        bars = self.ax_hist.bar(self.channel_names, delta_powers,
                                color='skyblue', edgecolor='navy')
        self.ax_hist.set_title('Мощность дельта-ритма по каналам', fontweight='bold', fontsize=14)
        self.ax_hist.set_ylabel('Мощность (мкВ²/Гц)', fontsize=12)
        self.ax_hist.grid(True, alpha=0.3)

        # Добавляем значения на столбцы гистограммы
        for bar, power in zip(bars, delta_powers):
            self.ax_hist.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + max(delta_powers) * 0.01,
                              f'{power:.2f}', ha='center', va='bottom', fontsize=11)

        # Графики 2-7: СПМ для каждого канала
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        for i, (ax, name, color) in enumerate(zip(self.ax_psd, self.channel_names, colors)):
            if all_psd_data[i] is not None and all_freqs_data[i] is not None:
                # Рисуем график СПМ
                ax.plot(all_freqs_data[i], all_psd_data[i], color=color, linewidth=1)
                ax.set_xlim(0, 8)

                # Безопасное вычисление максимума для оси Y
                mask = all_freqs_data[i] <= 8
                if len(all_psd_data[i][mask]) > 0:
                    y_max = max(all_psd_data[i][mask]) * 1.1
                    ax.set_ylim(0, y_max)

                # Подсвечиваем дельта-диапазон
                delta_mask = (all_freqs_data[i] >= 0.5) & (all_freqs_data[i] <= 4.0)
                ax.fill_between(all_freqs_data[i][delta_mask],
                                all_psd_data[i][delta_mask],
                                alpha=0.3, color=color)

                ax.set_title(f'{name} - Δ: {delta_powers[i]:.2f}',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('Частота (Гц)', fontsize=10)
                ax.set_ylabel('СПМ (мкВ²/Гц)', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Добавляем вертикальные линии для границ дельта-ритма
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
                ax.axvline(x=4.0, color='gray', linestyle='--', alpha=0.7)
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