import jax
import jax.numpy as jnp
import flax
import optax

print(f"JAX версия: {jax.__version__}")
print(f"Flax версия: {flax.__version__}")
print(f"Optax версия: {optax.__version__}")

# Проверка доступных устройств
print(f"Доступные устройства: {jax.devices()}")

# Простой тест
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
y = jnp.dot(x, x.T)
print(f"Тест матричного умножения прошел: {y.shape}")
