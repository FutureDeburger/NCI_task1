import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class EEGAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор ЭЭГ — Дельта ритм и синусоиды")
        self.root.geometry("1600x1200")

        # Основные переменные
        self.data = None
        self.channel_names = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
        self.fs = 5000
        self.current_view = 'analysis'

        self.freq_bands = {
            'delta': (0.5, 3.0),
            'theta': (4.0, 7.0),
            'alpha': (8.0, 13.0),
            'beta': (15.0, 30.0),
            'gamma': (30.0, 45.0)
        }

        self.create_widgets()

    # ------------------------------------------
    #  UI компоненты
    # ------------------------------------------
    def create_widgets(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая часть (графики)
        left_frame = tk.Frame(main_paned)
        main_paned.add(left_frame, width=1000)

        # Правая часть (результаты)
        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, width=600)

        # Заголовок
        tk.Label(left_frame, text="Анализатор ЭЭГ — Дельта ритм", font=("Arial", 16, "bold")).pack(pady=15)

        # Кнопки
        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)

        # Первая строка кнопок
        button_row1 = tk.Frame(button_frame)
        button_row1.pack(pady=5)

        load_btn = tk.Button(button_row1, text="ЗАГРУЗИТЬ ФАЙЛ .ASC",
                             command=self.load_file,
                             font=("Arial", 11, "bold"),
                             width=22, height=2, bg="#4CAF50", fg="white")
        load_btn.pack(side=tk.LEFT, padx=5)

        self.view_eeg_btn = tk.Button(button_row1, text="ПРОСМОТР ЭЭГ",
                                      command=self.view_raw_eeg,
                                      font=("Arial", 11, "bold"),
                                      width=22, height=2, bg="#607D8B", fg="white", state="disabled")
        self.view_eeg_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_delta_btn = tk.Button(button_row1, text="ДЕЛЬТА-РИТМ",
                                           command=lambda: self.analyze_data('delta'),
                                           font=("Arial", 11, "bold"),
                                           width=22, height=2, bg="#2196F3", fg="white", state="disabled")
        self.analyze_delta_btn.pack(side=tk.LEFT, padx=5)

        # Вторая строка кнопок
        button_row2 = tk.Frame(button_frame)
        button_row2.pack(pady=5)

        self.analyze_full_btn = tk.Button(button_row2, text="ПОЛНЫЙ СПЕКТР",
                                          command=lambda: self.analyze_data('full_spectrum'),
                                          font=("Arial", 11, "bold"),
                                          width=22, height=2, bg="#9C27B0", fg="white", state="disabled")
        self.analyze_full_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(button_row2, text="СОХРАНИТЬ РЕЗУЛЬТАТЫ",
                                  command=self.save_results,
                                  font=("Arial", 11, "bold"),
                                  width=22, height=2, bg="#FF9800", fg="white", state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(button_row2, text="ВЕРНУТЬСЯ К АНАЛИЗУ",
                                  command=self.show_analysis_view,
                                  font=("Arial", 11, "bold"),
                                  width=22, height=2, bg="#795548", fg="white", state="disabled")
        self.back_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(left_frame, text="Файл не загружен", font=("Arial", 12))
        self.file_label.pack(pady=5)

        self.create_plot_area(left_frame)
        self.create_results_area(right_frame)

    # ------------------------------------------
    #  Область графиков
    # ------------------------------------------
    def create_plot_area(self, parent):
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig = plt.figure(figsize=(12, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ------------------------------------------
    #  Область результатов
    # ------------------------------------------
    def create_results_area(self, parent):
        results_frame = tk.Frame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.results_label = tk.Label(results_frame, text="РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:",
                                      font=("Arial", 14, "bold"))
        self.results_label.pack(anchor='w', pady=(0, 10))
        self.results_text = tk.Text(results_frame, height=35, width=80, font=("Courier", 12),
                                    wrap=tk.WORD, bg="#f8f9fa", relief=tk.SUNKEN, bd=2, padx=10, pady=10)
        self.results_text.pack(fill='both', expand=True)

    # ------------------------------------------
    #  Загрузка файла
    # ------------------------------------------
    def load_file(self):
        path = filedialog.askopenfilename(title="Выберите файл .asc",
                                          filetypes=[("ASC files", "*.asc"), ("All files", "*.*")])
        if not path:
            return
        try:
            lines = [l.strip() for l in open(path, 'r', encoding='utf-8') if l.strip() and not l.startswith(';')]
            data = [list(map(float, l.split())) for l in lines]
            self.data = np.array(data)
            self.total_duration = len(self.data) / self.fs
            self.file_label.config(text=f"Загружен: {os.path.basename(path)}")

            for b in [self.analyze_delta_btn, self.analyze_full_btn, self.save_btn, self.view_eeg_btn, self.back_btn]:
                b.config(state="normal")

            messagebox.showinfo("Успех", f"Файл загружен!\nКаналов: {self.data.shape[1]}\n"
                                         f"Длительность: {self.total_duration:.2f} сек")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    # ------------------------------------------
    #  Просмотр исходной ЭЭГ
    # ------------------------------------------
    def view_raw_eeg(self):
        """Показывает исходный сигнал ЭЭГ (10 секунд по каждому каналу)"""
        if self.data is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите файл ЭЭГ.")
            return

        self.fig.clear()
        gs = plt.GridSpec(6, 1, figure=self.fig)
        t = np.arange(self.data.shape[0]) / self.fs

        for i, name in enumerate(self.channel_names):
            ax = self.fig.add_subplot(gs[i, 0])
            ax.plot(t, self.data[:, i], linewidth=0.8)
            ax.set_xlim(0, min(10, t[-1]))
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title("Исходный сигнал ЭЭГ (первые 10 секунд)")

        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------
    #  Возврат в анализ
    # ------------------------------------------
    def show_analysis_view(self):
        self.fig.clear()
        self.results_label.config(text="РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Готов к анализу. Выберите тип анализа.")
        self.canvas.draw()

    # ------------------------------------------
    #  Анализ данных
    # ------------------------------------------
    def compute_psd(self, signal, fs):
        N = len(signal)
        if N < 1000 or np.std(signal) < 1e-10:
            return None, None
        signal = signal - np.mean(signal)
        window = np.hanning(N)
        fft_result = np.fft.fft(signal * window, n=N)
        psd = (np.abs(fft_result) ** 2) / (fs * np.sum(window ** 2))
        freqs = np.fft.fftfreq(N, 1 / fs)
        mask = (freqs >= 0) & (freqs <= fs / 2)
        return freqs[mask], psd[mask]

    def compute_band_power(self, psd, freqs, f1, f2):
        mask = (freqs >= f1) & (freqs <= f2)
        return np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0

    def analyze_data(self, analysis_type):
        if self.data is None:
            return

        fs = self.fs
        powers, psd_all, freqs_all = [], [], []
        results = ""

        if analysis_type == 'delta':
            results += "АНАЛИЗ СПМ ДЕЛЬТА-РИТМА (0.5–3 Гц)\n" + "=" * 80 + "\n\n"
            rng = self.freq_bands['delta']
            self.results_label.config(text="РЕЗУЛЬТАТЫ ДЕЛЬТА-РИТМА:")
        else:
            results += "АНАЛИЗ ПОЛНОГО СПЕКТРА (0.5–45 Гц)\n" + "=" * 80 + "\n\n"
            rng = (0.5, 45.0)
            self.results_label.config(text="РЕЗУЛЬТАТЫ ПОЛНОГО СПЕКТРА:")

        for i, name in enumerate(self.channel_names):
            sig = self.data[:, i]
            freqs, psd = self.compute_psd(sig, fs)
            if freqs is None:
                powers.append(0)
                psd_all.append(None)
                freqs_all.append(None)
                results += f"{name}: нет данных\n"
                continue
            psd_all.append(psd)
            freqs_all.append(freqs)
            pwr = self.compute_band_power(psd, freqs, rng[0], rng[1])
            powers.append(pwr)
            results += f"{name}: {pwr:.6f} мкВ²/Гц\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.update_plots(powers, psd_all, freqs_all, analysis_type)

    # ------------------------------------------
    #  Обновление графиков
    # ------------------------------------------
    def update_plots(self, powers, psd_all, freqs_all, analysis_type):
        self.fig.clear()

        if analysis_type == 'delta':
            gs = plt.GridSpec(4, 3, figure=self.fig, height_ratios=[1.2, 1, 1, 1])
        else:
            gs = plt.GridSpec(3, 3, figure=self.fig, height_ratios=[1.2, 1, 1])

        ax_hist = self.fig.add_subplot(gs[0, :])
        ax_psd = [self.fig.add_subplot(gs[1 + i // 3, i % 3]) for i in range(6)]

        # Гистограмма мощностей
        ax_hist.bar(self.channel_names, powers, color='skyblue', edgecolor='navy')
        ax_hist.set_title("Мощность дельта-ритма по каналам" if analysis_type == 'delta'
                          else "Мощность полного спектра", fontsize=13, fontweight='bold')
        ax_hist.set_ylabel("мкВ²/Гц")
        ax_hist.grid(True, alpha=0.3)

        # СПМ графики
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (ax, name, color) in enumerate(zip(ax_psd, self.channel_names, colors)):
            if psd_all[i] is None:
                ax.text(0.5, 0.5, "Нет данных", ha='center', va='center')
                continue
            freqs, psd = freqs_all[i], psd_all[i]
            ax.plot(freqs, psd, color=color, linewidth=1)
            ax.set_xlim(0, 6 if analysis_type == 'delta' else 50)
            ax.set_xlabel("Частота (Гц)")
            ax.set_ylabel("СПМ (мкВ²/Гц)")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        # Синусоиды
        if analysis_type == 'delta':
            t = np.linspace(0, 2, 1000)
            ax_sin = [self.fig.add_subplot(gs[3, i]) for i in range(3)]
            for i in range(6):
                if psd_all[i] is None:
                    continue
                freqs, psd = freqs_all[i], psd_all[i]
                mask = (freqs >= self.freq_bands['delta'][0]) & (freqs <= self.freq_bands['delta'][1])
                if not np.any(mask):
                    continue
                f_peak = freqs[mask][np.argmax(psd[mask])]
                y = np.sin(2 * np.pi * f_peak * t)
                ax = ax_sin[i // 2]
                ax.plot(t, y, label=f"{self.channel_names[i]} ({f_peak:.2f} Гц)")
                ax.set_xlabel("Время (с)")
                ax.set_ylabel("Амплитуда")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
            for ax in ax_sin:
                ax.set_title("Синусоиды дельта-ритма (0.5–3 Гц)", fontsize=11, fontweight='bold')

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    # ------------------------------------------
    #  Сохранение
    # ------------------------------------------
    def save_results(self):
        path = filedialog.asksaveasfilename(title="Сохранить результаты",
                                            defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt")])
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.results_text.get(1.0, tk.END))
        self.fig.savefig(path.replace('.txt', '_plot.png'), dpi=300, bbox_inches='tight')
        messagebox.showinfo("Сохранено", f"Результаты сохранены:\n{path}")


# ------------------------------------------
#  Запуск приложения
# ------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalyzerApp(root)
    root.mainloop()
