import numpy as np
import matplotlib.pyplot as plt

# Загружаем данные
data = np.loadtxt('ткачев.asc', skiprows=4)  # Пропускаем 4 строки заголовка

# Названия каналов из вашего файла
channel_names = ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
fs = 5000  # Частота дискретизации

# Создаем список для хранения результатов
delta_powers = []

# Анализируем КАЖДЫЙ из 6 каналов
for channel_idx in range(6):  # от 0 до 5 - все 6 каналов!
    print(f"\n=== Анализ канала {channel_names[channel_idx]} (столбец {channel_idx}) ===")

    # Берем данные для текущего канала
    eeg_signal = data[:, channel_idx]

    # Убираем нули (если есть в начале)
    eeg_signal = eeg_signal[eeg_signal != 0]

    if len(eeg_signal) == 0:
        print("⚠️  Нет данных в канале!")
        delta_powers.append(0)
        continue

    N = len(eeg_signal)

    # 1. Прямое преобразование Фурье
    fft_result = np.fft.fft(eeg_signal)

    # 2. Вычисляем СПМ по формуле: Re² + Im²
    psd = np.real(fft_result) ** 2 + np.imag(fft_result) ** 2
    psd = psd / N  # Нормализация

    # 3. Частотная ось
    frequencies = np.fft.fftfreq(N, 1 / fs)

    # 4. Выделяем дельта-ритм (0.5-4 Гц)
    positive_idx = frequencies >= 0
    freqs_positive = frequencies[positive_idx]
    psd_positive = psd[positive_idx]

    delta_mask = (freqs_positive >= 0.5) & (freqs_positive <= 4.0)
    delta_freqs = freqs_positive[delta_mask]
    delta_psd = psd_positive[delta_mask]

    # 5. Мощность дельта-ритма
    delta_power = np.trapz(delta_psd, delta_freqs)
    delta_powers.append(delta_power)

    print(f"Мощность дельта-ритма: {delta_power:.6f} мкВ²/Гц")

# Выводим сводную таблицу результатов
print("\n" + "=" * 50)
print("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 50)
for i, (name, power) in enumerate(zip(channel_names, delta_powers)):
    print(f"{name}: {power:.6f} мкВ²/Гц")

# Создаем большую фигуру для всех каналов
plt.figure(figsize=(12, 8))

for channel_idx in range(6):
    eeg_signal = data[:, channel_idx]
    eeg_signal = eeg_signal[eeg_signal != 0]

    if len(eeg_signal) == 0:
        continue

    # Вычисляем СПМ для визуализации
    fft_result = np.fft.fft(eeg_signal)
    psd = (np.real(fft_result) ** 2 + np.imag(fft_result) ** 2) / len(eeg_signal)
    frequencies = np.fft.fftfreq(len(eeg_signal), 1 / fs)

    positive_idx = frequencies >= 0
    freqs_positive = frequencies[positive_idx]
    psd_positive = psd[positive_idx]

    # Рисуем график для каждого канала
    plt.subplot(2, 3, channel_idx + 1)  # 2 строки, 3 столбца

    plt.plot(freqs_positive, psd_positive, 'b-', linewidth=1)

    # Подсвечиваем дельта-ритм
    delta_mask = (freqs_positive >= 0.5) & (freqs_positive <= 4.0)
    delta_freqs = freqs_positive[delta_mask]
    delta_psd = psd_positive[delta_mask]

    plt.fill_between(delta_freqs, delta_psd, alpha=0.5, color='red')
    plt.xlim(0, 8)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('СПМ')
    plt.title(f'{channel_names[channel_idx]}\nМощность: {delta_powers[channel_idx]:.4f}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('СПЕКТРАЛЬНАЯ ПЛОТНОСТЬ МОЩНОСТИ - ВСЕ 6 КАНАЛОВ', fontsize=16, y=1.02)
plt.show()


# Сравнительная гистограмма
plt.figure(figsize=(12, 6))
bars = plt.bar(channel_names, delta_powers, color=['skyblue', 'lightcoral', 'lightgreen',
                                                  'gold', 'plum', 'lightblue'])

# Добавляем значения на столбцы
for bar, power in zip(bars, delta_powers):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(delta_powers)*0.01,
             f'{power:.4f}', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Мощность дельта-ритма (мкВ²/Гц)')
plt.title('СРАВНЕНИЕ МОЩНОСТИ ДЕЛЬТА-РИТМА ПО КАНАЛАМ')
plt.grid(True, alpha=0.3)

# Подписываем области
plt.text(0.5, 0.95, 'ТЕМЕННАЯ ОБЛАСТЬ', transform=plt.gca().transAxes,
         ha='center', fontweight='bold', color='darkred')
plt.text(0.5, 0.90, 'ЗАТЫЛОЧНАЯ ОБЛАСТЬ', transform=plt.gca().transAxes,
         ha='center', fontweight='bold', color='darkblue')

plt.show()