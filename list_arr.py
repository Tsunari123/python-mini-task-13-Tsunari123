import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_field(filename):
    with open(filename, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    return data


def save_field(field, filename):
    with open(filename, 'wb') as f:
        np.save(f, field)


def generate_field(size=1024, seed=42):
    np.random.seed(seed)
    return np.random.choice([0, 1], size=(size, size))


class LifeGamePython:

    def __init__(self, field):
        self.field = [list(row) for row in field]
        self.size = len(field)
        self.iteration = 0

    def update(self):
        new_field = [[0] * self.size for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                neighbors = 0
                for di in [-1, 0, 1]:
                    dj: int
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            neighbors += self.field[ni][nj]

                if self.field[i][j] == 1:
                    new_field[i][j] = 1 if neighbors in [2, 3] else 0
                else:
                    new_field[i][j] = 1 if neighbors == 3 else 0

        self.field = new_field
        self.iteration += 1

    def count_alive(self):
        return sum(sum(row) for row in self.field)


class LifeGameNumPy:

    def __init__(self, field):
        self.field = field.copy()
        self.size = len(field)
        self.iteration = 0

    def update(self):
        padded = np.pad(self.field, 1, mode='constant', constant_values=0)

        neighbors = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )

        birth = (self.field == 0) & (neighbors == 3)
        survive = (self.field == 1) & ((neighbors == 2) | (neighbors == 3))
        self.field = np.where(birth | survive, 1, 0)
        self.iteration += 1

    def count_alive(self):
        return np.sum(self.field)


def visualize_results(initial_field, python_field, numpy_field, python_time, numpy_time, py_alive, np_alive):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(initial_field, cmap='binary')
    axes[0, 0].set_title(f'Начальное состояние\nЖивых клеток: {np.sum(initial_field)}')
    axes[0, 0].axis('off')

    python_field_np = np.array(python_field)
    axes[0, 1].imshow(python_field_np, cmap='binary')
    axes[0, 1].set_title(f'Python версия (после 128 итераций)\nВремя: {python_time:.2f} сек\nЖивых клеток: {py_alive}')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(numpy_field, cmap='binary')
    axes[1, 0].set_title(f'NumPy версия (после 128 итераций)\nВремя: {numpy_time:.2f} сек\nЖивых клеток: {np_alive}')
    axes[1, 0].axis('off')

    diff = np.abs(python_field_np - numpy_field)
    diff_img = axes[1, 1].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Разница между версиями\nРазных клеток: {np.sum(diff)}')
    axes[1, 1].axis('off')
    plt.colorbar(diff_img, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Сравнение Python и NumPy версий игры "Жизнь"\nУскорение: {python_time / numpy_time:.1f}x',
                 fontsize=14)
    plt.tight_layout()
    plt.show()


def create_animation(field, iterations=50, interval=100):
    game = LifeGameNumPy(field)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(game.field, cmap='binary')
    ax1.set_title('Игра "Жизнь" - Текущее состояние')
    ax1.axis('off')

    alive_counts = [game.count_alive()]
    line, = ax2.plot(alive_counts, 'b-', linewidth=2)
    ax2.set_title('Количество живых клеток')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Живые клетки')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, iterations)
    ax2.set_ylim(0, field.size)

    def update(frame):
        game.update()
        current_alive = game.count_alive()

        im.set_array(game.field)
        ax1.set_title(f'Итерация: {game.iteration}')

        alive_counts.append(current_alive)
        line.set_data(range(len(alive_counts)), alive_counts)

        ax2.set_xlim(0, max(iterations, len(alive_counts)))
        ax2.set_ylim(0, max(field.size, max(alive_counts) * 1.1))

        return im, line

    ani = FuncAnimation(fig, update, frames=iterations, interval=interval, blit=False)

    plt.tight_layout()
    plt.show()

    return ani


def run_comparison(field, iterations=128):
    game_py = LifeGamePython(field)

    start_time = time.time()
    for i in range(iterations):
        game_py.update()
    py_time = time.time() - start_time
    py_alive = game_py.count_alive()

    game_np = LifeGameNumPy(field)

    start_time = time.time()
    for i in range(iterations):
        game_np.update()
    np_time = time.time() - start_time
    np_alive = game_np.count_alive()

    py_field_np = np.array(game_py.field)
    arrays_equal = np.array_equal(py_field_np, game_np.field)

    return py_time, np_time, py_alive, np_alive, arrays_equal, game_py.field, game_np.field


def main():
    try:
        field = read_field("field_1024.npy")
    except FileNotFoundError:
        field = generate_field(1024, 42)
        save_field(field, "field_1024.npy")

    print("Поле 1024x1024")
    print("Итераций 128")

    py_time, np_time, py_alive, np_alive, arrays_equal, py_field, np_field = run_comparison(field, 128)

    print(f"Python версия: {py_time:.2f} сек, {py_alive} живых клеток")
    print(f"NumPy версия:  {np_time:.2f} сек, {np_alive} живых клеток")
    print(f"Ускорение NumPy {py_time / np_time:.1f}x")

    visualize_results(field, py_field, np_field, py_time, np_time, py_alive, np_alive)

    try:
        small_field = read_field("field_256.npy")
    except FileNotFoundError:
        small_field = generate_field(256, 42)
        save_field(small_field, "field_256.npy")

    create_animation(small_field, iterations=50, interval=100)


if __name__ == "__main__":
    main()