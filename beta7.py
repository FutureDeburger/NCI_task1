import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class EEGAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор ЭЭГ")
        self.root.geometry("1600x1200")

        self.data = None
        self.channel_names = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
        self.current_view = 'analysis'
        self.current_channel = 0
        self.fs = 5000
        self.eeg_display_seconds = 10
        self.eeg_start_time = 0

        # Диапазоны частот
        self.freq_bands = {
            'delta': (0.5, 3.0),
            'theta': (4.0, 7.0),
            'alpha': (8.0, 13.0),
            'beta': (15.0, 30.0),
            'gamma': (30.0, 45.0)
        }

        self.create_widgets()

    def create_widgets(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = tk.Frame(main_paned)
        main_paned.add(left_frame, width=1000)

        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, width=600)

        title_label = tk.Label(left_frame, text="Анализ СПМ дельта-ритма", font=("Arial", 16, "bold"))
        title_label.pack(pady=15)

        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)

        button_row1 = tk.Frame(button_frame)
        button_row1.pack(pady=5)

        # Кнопки
        load_btn = tk.Button(button_row1, text="ЗАГРУЗИТЬ ФАЙЛ .ASC",
                             command=self.load_file, font=("Arial", 11, "bold"),
                             width=22, height=2, bg="#4CAF50", fg="white", activebackground="#45a049")
        load_btn.pack(side=tk.LEFT, padx=5)

        self.view_eeg_btn = tk.Button(button_row1, text="ПРОСМОТР ЭЭГ",
                                      command=self.view_raw_eeg, font=("Arial", 11, "bold"),
                                      width=22, height=2, bg="#607D8B", fg="white", activebackground="#546E7A",
                                      state="disabled")
        self.view_eeg_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_delta_btn = tk.Button(button_row1, text="ДЕЛЬТА-РИТМ",
                                           command=lambda: self.analyze_data('delta'), font=("Arial", 11, "bold"),
                                           width=22, height=2, bg="#2196F3", fg="white", activebackground="#0b7dda",
                                           state="disabled")
        self.analyze_delta_btn.pack(side=tk.LEFT, padx=5)

        self.view_delta_btn = tk.Button(button_row1, text="Δ-СИНУСОИДЫ",
                                        command=self.view_delta_sinusoids, font=("Arial", 11, "bold"),
                                        width=22, height=2, bg="#FF5722", fg="white", activebackground="#E64A19",
                                        state="disabled")
        self.view_delta_btn.pack(side=tk.LEFT, padx=5)

        button_row2 = tk.Frame(button_frame)
        button_row2.pack(pady=5)

        self.analyze_full_btn = tk.Button(button_row2, text="ПОЛНЫЙ СПЕКТР",
                                          command=lambda: self.analyze_data('full_spectrum'),
                                          font=("Arial", 11, "bold"), width=22, height=2,
                                          bg="#9C27B0", fg="white", activebackground="#7B1FA2", state="disabled")
        self.analyze_full_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(button_row2, text="СОХРАНИТЬ РЕЗУЛЬТАТЫ",
                                  command=self.save_results, font=("Arial", 11, "bold"), width=22, height=2,
                                  bg="#FF9800", fg="white", state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(button_row2, text="ВЕРНУТЬСЯ К АНАЛИЗУ",
                                  command=self.show_analysis_view, font=("Arial", 11, "bold"), width=22, height=2,
                                  bg="#795548", fg="white", state="disabled")
        self.back_btn.pack(side=tk.LEFT, padx=5)

        # Фреймы управления просмотром ЭЭГ
        self.eeg_control_frame = tk.Frame(button_frame)
        channel_frame = tk.Frame(self.eeg_control_frame)
        channel_frame.pack(pady=5)
        tk.Label(channel_frame, text="Каналы:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.channel_buttons = []
        for i, channel in enumerate(self.channel_names):
            btn = tk.Button(channel_frame, text=channel,
                            command=lambda idx=i: self.switch_channel(idx),
                            font=("Arial", 9, "bold"), width=8, height=1, bg="#E0E0E0", fg="black",
                            state="disabled")
            btn.pack(side=tk.LEFT, padx=2)
            self.channel_buttons.append(btn)

        time_frame = tk.Frame(self.eeg_control_frame)
        time_frame.pack(pady=5)
        tk.Label(time_frame, text="Окно просмотра:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.time_buttons = []
        time_options = [5, 10, 30, 60]
        for seconds in time_options:
            btn = tk.Button(time_frame, text=f"{seconds} сек",
                            command=lambda s=seconds: self.set_time_window(s),
                            font=("Arial", 9), width=10, height=1, bg="#F5F5F5", fg="black", state="disabled")
            btn.pack(side=tk.LEFT, padx=2)
            self.time_buttons.append(btn)

        scroll_frame = tk.Frame(self.eeg_control_frame)
        scroll_frame.pack(pady=5)
        self.scroll_left_btn = tk.Button(scroll_frame, text="◀ НАЗАД", command=self.scroll_left,
                                         font=("Arial", 9), width=12, height=1, bg="#BBDEFB", fg="black",
                                         state="disabled")
        self.scroll_left_btn.pack(side=tk.LEFT, padx=2)
        self.scroll_right_btn = tk.Button(scroll_frame, text="ВПЕРЕД ▶", command=self.scroll_right,
                                          font=("Arial", 9), width=12, height=1, bg="#BBDEFB", fg="black",
                                          state="disabled")
        self.scroll_right_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_out_btn = tk.Button(scroll_frame, text="− УВЕЛИЧИТЬ ОКНО", command=self.zoom_out,
                                      font=("Arial", 9), width=17, height=1, bg="#C8E6C9", fg="black", state="disabled")
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_in_btn = tk.Button(scroll_frame, text="+ УМЕНЬШИТЬ ОКНО", command=self.zoom_in,
                                     font=("Arial", 9), width=17, height=1, bg="#FFCDD2", fg="black", state="disabled")
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)

        self.file_label = tk.Label(left_frame, text="Файл не загружен", font=("Arial", 12))
        self.file_label.pack(pady=5)

        self.create_plot_area(left_frame)
        self.create_results_area(right_frame)

    def create_plot_area(self, parent):
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.fig = plt.figure(figsize=(12, 9))
        self.setup_analysis_grid()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_analysis_grid(self):
        self.fig.clear()
        self.gs = plt.GridSpec(3, 3, figure=self.fig, height_ratios=[1.2, 1, 1])
        self.ax_hist = self.fig.add_subplot(self.gs[0, :])
        self.ax_psd = []
        for i in range(6):
            row = 1 + i // 3
            col = i % 3
            self.ax_psd.append(self.fig.add_subplot(self.gs[row, col]))
        self.fig.tight_layout(pad=3.0)

    def setup_eeg_grid(self):
        self.fig.clear()
        self.ax_eeg = self.fig.add_subplot(111)
        self.ax_minimap = self.fig.add_axes([0.1, 0.85, 0.8, 0.05])
        self.fig.tight_layout(pad=3.0)

    def create_results_area(self, parent):
        results_frame = tk.Frame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.results_label = tk.Label(results_frame, text="РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:", font=("Arial", 14, "bold"))
        self.results_label.pack(anchor='w', pady=(0, 10))
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True)
        self.results_text = tk.Text(text_frame, height=35, width=80, font=("Courier", 12), wrap=tk.WORD,
                                    bg="#f8f9fa", relief=tk.SUNKEN, bd=2, padx=10, pady=10)
        self.results_text.pack(fill='both', expand=True)
        self.results_text.config(state="normal")
        results_buttons_frame = tk.Frame(results_frame)
        results_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        clear_btn = tk.Button(results_buttons_frame, text="ОЧИСТИТЬ РЕЗУЛЬТАТЫ", command=self.clear_results,
                              font=("Arial", 10, "bold"), width=22, height=1, bg="#F44336", fg="white")
        clear_btn.pack(side=tk.LEFT, padx=5)
        copy_btn = tk.Button(results_buttons_frame, text="КОПИРОВАТЬ РЕЗУЛЬТАТЫ", command=self.copy_results,
                             font=("Arial", 10, "bold"), width=22, height=1, bg="#2196F3", fg="white")
        copy_btn.pack(side=tk.LEFT, padx=5)

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)

    def copy_results(self):
        results = self.results_text.get(1.0, tk.END)
        if results.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(results)
            messagebox.showinfo("Успех", "Результаты скопированы в буфер обмена!")
        else:
            messagebox.showwarning("Внимание", "Нет данных для копирования")

    def load_file(self):
        file_path = filedialog.askopenfilename(title="Выберите файл .asc",
                                               filetypes=[("ASC files", "*.asc"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                data_lines = [line.strip() for line in lines if not line.startswith(';') and line.strip()]
                data = [[float(x) for x in line.split()] for line in data_lines]
                self.data = np.array(data)
                self.total_duration = len(self.data) / self.fs

                self.file_label.config(text=f"Загружен: {os.path.basename(file_path)}")
                self.analyze_delta_btn.config(state="normal")
                self.analyze_full_btn.config(state="normal")
                self.save_btn.config(state="normal")
                self.view_eeg_btn.config(state="normal")
                self.view_delta_btn.config(state="normal")
                self.back_btn.config(state="normal")
                for btn in self.channel_buttons:
                    btn.config(state="normal", bg="#E0E0E0")
                for btn in self.time_buttons:
                    btn.config(state="normal")
                self.scroll_left_btn.config(state="normal")
                self.scroll_right_btn.config(state="normal")
                self.zoom_in_btn.config(state="normal")
                self.zoom_out_btn.config(state="normal")
                self.channel_buttons[0].config(bg="#2196F3", fg="white")

                messagebox.showinfo("Успех",
                                    f"Файл загружен!\nКаналов: {self.data.shape[1]}\nОтсчетов: {self.data.shape[0]}\nДлительность: {self.total_duration:.2f} сек")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    # --- Вспомогательные функции СПМ ---
    def compute_psd(self, signal, fs):
        N = len(signal)
        if N < 1000 or np.std(signal) < 1e-10:
            return None, None
        signal = signal - np.mean(signal)
        window = np.hanning(N)
        fft_result = np.fft.fft(signal * window, n=N)
        psd = (np.abs(fft_result) ** 2) / (fs * np.sum(window ** 2))
        freqs = np.fft.fftfreq(N, 1 / fs)
        positive_idx = (freqs >= 0) & (freqs <= fs / 2)
        return freqs[positive_idx], psd[positive_idx]

    def compute_band_power(self, psd, freqs, f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        return np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0

    # --- Метод Δ-синусоид ---
    def view_delta_sinusoids(self):
        """Отображение синусоид дельта-ритма по каждому каналу"""
        if self.data is None:
            return

        try:
            self.current_view = 'eeg'  # Можно использовать ту же сетку для графиков
            self.fig.clear()
            self.ax_sin = []
            self.gs = plt.GridSpec(2, 3, figure=self.fig)  # 2 строки по 3 канала

            t = np.arange(0, 2, 1 / self.fs)  # 2 секунды для синусоиды

            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

            for i, name in enumerate(self.channel_names):
                ax = self.fig.add_subplot(self.gs[i // 3, i % 3])
                self.ax_sin.append(ax)

                signal = self.data[:, i]
                freqs, psd = self.compute_psd(signal, self.fs)

                if freqs is not None and psd is not None:
                    # Маска дельта-ритма
                    delta_mask = (freqs >= self.freq_bands['delta'][0]) & (freqs <= self.freq_bands['delta'][1])
                    delta_freqs = freqs[delta_mask]
                    delta_psd = psd[delta_mask]

                    if len(delta_psd) > 0:
                        # Частота максимальной мощности
                        peak_idx = np.argmax(delta_psd)
                        peak_freq = delta_freqs[peak_idx]
                        peak_power = delta_psd[peak_idx]

                        # Амплитуда синусоиды = sqrt(мощность)
                        amplitude = np.sqrt(peak_power)

                        y = amplitude * np.sin(2 * np.pi * peak_freq * t)
                        ax.plot(t, y, color=colors[i], linewidth=2)
                        ax.set_title(f"{name} - {peak_freq:.2f} Гц", fontsize=12, fontweight='bold')
                        ax.set_xlabel('Время (сек)')
                        ax.set_ylabel('Амплитуда (мкВ)')
                        ax.grid(True)
                    else:
                        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить синусоиды:\n{str(e)}")

    # --- Остальной функционал (просмотр ЭЭГ, анализ и т.д.) ---
    def view_raw_eeg(self):
        if self.data is None:
            return
        self.current_view = 'eeg'
        self.eeg_start_time = 0
        self.setup_eeg_grid()
        self.eeg_control_frame.pack(pady=10)
        self.update_eeg_display()

    def update_eeg_display(self):
        if self.data is None or self.current_view != 'eeg':
            return
        self.ax_eeg.clear()
        self.ax_minimap.clear()
        channel_data = self.data[:, self.current_channel]
        time = np.arange(len(channel_data)) / self.fs
        start_idx = int(self.eeg_start_time * self.fs)
        end_idx = int((self.eeg_start_time + self.eeg_display_seconds) * self.fs)
        end_idx = min(end_idx, len(channel_data))
        display_data = channel_data[start_idx:end_idx]
        display_time = time[start_idx:end_idx]
        self.ax_eeg.plot(display_time, display_data, color='#2196F3', linewidth=1)
        self.ax_eeg.set_title(f'ЭЭГ - Канал {self.channel_names[self.current_channel]}',
                              fontweight='bold', fontsize=14)
        self.ax_eeg.set_ylabel('Амплитуда (мкВ)', fontsize=12)
        self.ax_eeg.set_xlabel('Время (секунды)', fontsize=12)
        self.ax_eeg.grid(True, alpha=0.3)
        y_margin = (np.max(display_data) - np.min(display_data)) * 0.1
        if y_margin == 0:
            y_margin = 1
        self.ax_eeg.set_ylim(np.min(display_data) - y_margin, np.max(display_data) + y_margin)
        self.ax_minimap.plot(time, channel_data, color='gray', linewidth=0.5, alpha=0.7)
        window_start = self.eeg_start_time
        window_end = self.eeg_start_time + self.eeg_display_seconds
        self.ax_minimap.axvspan(window_start, window_end, alpha=0.3, color='red')
        self.ax_minimap.set_xlim(0, time[-1])
        self.ax_minimap.set_ylim(np.min(channel_data), np.max(channel_data))
        self.ax_minimap.set_ylabel('Амплитуда', fontsize=8)
        self.ax_minimap.set_xlabel('Время', fontsize=8)
        self.canvas.draw()

    def switch_channel(self, idx):
        self.current_channel = idx
        for i, btn in enumerate(self.channel_buttons):
            if i == idx:
                btn.config(bg="#2196F3", fg="white")
            else:
                btn.config(bg="#E0E0E0", fg="black")
        self.update_eeg_display()

    def set_time_window(self, seconds):
        self.eeg_display_seconds = seconds
        self.update_eeg_display()

    def scroll_left(self):
        self.eeg_start_time = max(0, self.eeg_start_time - self.eeg_display_seconds)
        self.update_eeg_display()

    def scroll_right(self):
        self.eeg_start_time = min(self.total_duration - self.eeg_display_seconds,
                                  self.eeg_start_time + self.eeg_display_seconds)
        self.update_eeg_display()

    def zoom_in(self):
        if self.eeg_display_seconds > 1:
            self.eeg_display_seconds /= 2
        self.update_eeg_display()

    def zoom_out(self):
        if self.eeg_display_seconds < self.total_duration:
            self.eeg_display_seconds *= 2
        self.update_eeg_display()

    def analyze_data(self, mode):
        if self.data is None:
            return
        self.current_view = 'analysis'
        self.setup_analysis_grid()
        if mode == 'delta':
            self.ax_hist.clear()
            delta_powers = []
            for i, ch in enumerate(self.channel_names):
                freqs, psd = self.compute_psd(self.data[:, i], self.fs)
                if freqs is not None:
                    power = self.compute_band_power(psd, freqs, *self.freq_bands['delta'])
                    delta_powers.append(power)
                else:
                    delta_powers.append(0)
            self.ax_hist.bar(self.channel_names, delta_powers, color='green')
            self.ax_hist.set_title('Мощность дельта-ритма по каналам', fontweight='bold')
            self.ax_hist.set_ylabel('Мощность')
        elif mode == 'full_spectrum':
            for i, ch in enumerate(self.channel_names):
                freqs, psd = self.compute_psd(self.data[:, i], self.fs)
                self.ax_psd[i].clear()
                if freqs is not None:
                    self.ax_psd[i].plot(freqs, psd, color='blue')
                    self.ax_psd[i].set_title(f'{ch} - Полный спектр', fontweight='bold')
                    self.ax_psd[i].set_xlabel('Гц')
                    self.ax_psd[i].set_ylabel('Мощность')
                    self.ax_psd[i].grid(True)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_results(self):
        results = self.results_text.get(1.0, tk.END)
        if results.strip():
            file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                     filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(results)
                messagebox.showinfo("Успех", f"Результаты сохранены в {file_path}")
        else:
            messagebox.showwarning("Внимание", "Нет данных для сохранения")

    def show_analysis_view(self):
        self.current_view = 'analysis'
        self.setup_analysis_grid()
        self.canvas.draw()
        self.eeg_control_frame.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAnalyzerApp(root)
    root.mainloop()
