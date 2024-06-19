import jax

def check_jax():
    try:
        print(f"JAX version: {jax.__version__}")
    except ImportError as e:
        print(f"Error importing JAX: {e}")

if __name__ == "__main__":
    check_jax()
