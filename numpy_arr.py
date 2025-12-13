import numpy as np
import pickle


def generate_fields():
    np.random.seed(42)
    field_1024 = np.random.choice([0, 1], size=(1024, 1024))

    with open('initial_field_1024.pkl', 'wb') as f:
        pickle.dump({
            'grid': field_1024,
            'size': 1024,
            'seed': 42,
            'density': np.sum(field_1024) / (1024 * 1024),
            'alive_cells': np.sum(field_1024)
        }, f)

    np.random.seed(42)
    field_256 = np.random.choice([0, 1], size=(256, 256))

    with open('initial_field_256.pkl', 'wb') as f:
        pickle.dump({
            'grid': field_256,
            'size': 256,
            'seed': 42,
            'density': np.sum(field_256) / (256 * 256),
            'alive_cells': np.sum(field_256)
        }, f)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(field_1024, cmap='binary', interpolation='nearest')
    ax1.set_title(f'Поле 1024x1024\nЖивых клеток: {np.sum(field_1024)}')
    ax1.axis('off')

    ax2.imshow(field_256, cmap='binary', interpolation='nearest')
    ax2.set_title(f'Поле 256x256\nЖивых клеток: {np.sum(field_256)}')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('initial_fields.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    generate_fields()