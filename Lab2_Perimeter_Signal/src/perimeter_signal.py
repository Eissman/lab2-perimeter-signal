"""Лабораторная работа № 2: приём и предварительная обработка сигнала датчика.

Источник данных в учебном проекте — программная модель датчика вибрации
ограждения. Модель можно заменить адаптером реального датчика или API, не
меняя блок буферизации и сглаживания.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Sequence


# Вариант 1 из методических указаний лабораторной работы:
AMPLITUDES = (1.0, 0.5, 0.25)
FREQUENCIES_HZ = (0.1, 0.1, 0.4)


@dataclass(frozen=True)
class RawSample:
    """Один принятый отсчёт до предварительной обработки."""

    index: int
    time_s: float
    value: float


@dataclass(frozen=True)
class ProcessedSample:
    """Отсчёт после обработки EMA."""

    index: int
    time_s: float
    raw_value: float
    ema_value: float


class RingBuffer:
    """Ограниченный буфер последних N отсчётов датчика."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Размер буфера должен быть положительным")
        self._items: Deque[RawSample] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._items.maxlen or 0

    def append(self, sample: RawSample) -> None:
        self._items.append(sample)

    def items(self) -> tuple[RawSample, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


class ExponentialMovingAverage:
    """Экспоненциальное скользящее среднее EMA.

    alpha = 2 / (N + 1)
    EMA[t] = alpha * x[t] + (1 - alpha) * EMA[t-1]
    Первый выходной отсчёт инициализируется первым входным значением.
    """

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("Период сглаживания должен быть положительным")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._previous: float | None = None

    def update(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("На вход фильтра поступило нечисловое или бесконечное значение")
        if self._previous is None:
            result = value
        else:
            result = self.alpha * value + (1.0 - self.alpha) * self._previous
        self._previous = result
        return result


def harmonic_signal(
    time_s: float,
    amplitudes: Sequence[float] = AMPLITUDES,
    frequencies_hz: Sequence[float] = FREQUENCIES_HZ,
) -> float:
    """Вычисляет X(t) = sum(A_i * sin(2*pi*F_i*t))."""
    if len(amplitudes) != len(frequencies_hz):
        raise ValueError("Векторы амплитуд и частот должны иметь одинаковую длину")
    return sum(
        amplitude * math.sin(2.0 * math.pi * frequency * time_s)
        for amplitude, frequency in zip(amplitudes, frequencies_hz)
    )


def validate_raw_sample(sample: RawSample) -> None:
    """Первичная валидация принятого отсчёта перед передачей бизнес-логике."""
    if sample.index < 0:
        raise ValueError("Индекс отсчёта не может быть отрицательным")
    if not math.isfinite(sample.time_s) or sample.time_s < 0:
        raise ValueError("Временная метка должна быть конечным неотрицательным числом")
    if not math.isfinite(sample.value):
        raise ValueError("Значение сигнала должно быть конечным числом")


def receive_and_process(
    sample_rate_hz: float = 20.0,
    duration_s: float = 20.0,
    noise_std: float = 0.15,
    ema_period: int = 10,
    buffer_size: int = 100,
    seed: int = 42,
) -> tuple[list[ProcessedSample], RingBuffer, float]:
    """Имитирует приём дискретных отсчётов, буферизацию и EMA-сглаживание."""
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Частота выборки должна быть положительной")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Длительность сигнала должна быть положительной")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Уровень шума не может быть отрицательным")

    sample_count = int(round(sample_rate_hz * duration_s))
    if sample_count < 1:
        raise ValueError("Для заданных параметров не получается ни одного отсчёта")

    rng = random.Random(seed)
    buffer = RingBuffer(buffer_size)
    ema = ExponentialMovingAverage(ema_period)
    processed: list[ProcessedSample] = []
    sampling_interval = 1.0 / sample_rate_hz

    for index in range(sample_count):
        # Дискретизация: время каждого отсчёта задаётся частотой выборки.
        time_s = index * sampling_interval
        ideal_value = harmonic_signal(time_s)
        raw_value = ideal_value + (rng.gauss(0.0, noise_std) if noise_std else 0.0)
        raw_sample = RawSample(index=index, time_s=time_s, value=raw_value)

        # Приём и первичная проверка; сохраняем последние N сырых отсчётов.
        validate_raw_sample(raw_sample)
        buffer.append(raw_sample)

        # Предварительная обработка: сглаживание EMA до передачи в бизнес-логику.
        filtered_value = ema.update(raw_sample.value)
        processed.append(
            ProcessedSample(
                index=index,
                time_s=time_s,
                raw_value=raw_sample.value,
                ema_value=filtered_value,
            )
        )

    return processed, buffer, ema.alpha


def write_csv(samples: Iterable[ProcessedSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["index", "time_s", "raw_value", "ema_value"])
        for sample in samples:
            writer.writerow(
                [
                    sample.index,
                    f"{sample.time_s:.5f}",
                    f"{sample.raw_value:.6f}",
                    f"{sample.ema_value:.6f}",
                ]
            )


def write_plot(samples: Sequence[ProcessedSample], output_path: Path) -> bool:
    """Строит график. Возвращает False, если matplotlib не установлен."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = [sample.time_s for sample in samples]
    raw_values = [sample.raw_value for sample in samples]
    filtered_values = [sample.ema_value for sample in samples]

    plt.figure(figsize=(10, 5.5))
    plt.plot(times, raw_values, label="Сырой сигнал", linewidth=1.0, alpha=0.7)
    plt.plot(times, filtered_values, label="После EMA", linewidth=2.0)
    plt.title("Имитация сигнала датчика вибрации ограждения")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда сигнала")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация, буферизация и EMA-сглаживание сигнала датчика периметра"
    )
    parser.add_argument("--sample-rate", type=float, default=20.0, help="частота выборки в Гц")
    parser.add_argument("--duration", type=float, default=20.0, help="длительность моделирования в секундах")
    parser.add_argument("--noise", type=float, default=0.15, help="стандартное отклонение шума")
    parser.add_argument("--period", type=int, default=10, help="период EMA N")
    parser.add_argument("--buffer-size", type=int, default=100, help="ёмкость буфера сырых отсчётов")
    parser.add_argument("--seed", type=int, default=42, help="начальное значение генератора шума")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="папка результатов")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, buffer, alpha = receive_and_process(
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        noise_std=args.noise,
        ema_period=args.period,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    csv_path = args.out_dir / "perimeter_samples.csv"
    plot_path = args.out_dir / "perimeter_signal.png"
    write_csv(samples, csv_path)
    plot_created = write_plot(samples, plot_path)

    raw_mean = sum(sample.raw_value for sample in samples) / len(samples)
    ema_mean = sum(sample.ema_value for sample in samples) / len(samples)

    print("Имитация приёма сигнала завершена успешно.")
    print(f"Частота выборки: {args.sample_rate:g} Гц")
    print(f"Интервал между отсчётами: {1.0 / args.sample_rate:.5f} с")
    print(f"Длительность: {args.duration:g} с; принято отсчётов: {len(samples)}")
    print(f"EMA: период N={args.period}, alpha={alpha:.4f}")
    print(f"Буфер: {len(buffer)} из {buffer.capacity} последних сырых отсчётов")
    print(f"Среднее сырого сигнала: {raw_mean:.4f}")
    print(f"Среднее сигнала после EMA: {ema_mean:.4f}")
    print(f"CSV сохранён: {csv_path}")
    if plot_created:
        print(f"График сохранён: {plot_path}")
    else:
        print("График не создан: установи зависимости командой pip install -r requirements.txt")


if __name__ == "__main__":
    main()
