"""Adım 6 — TMAGym Gymnasium API uyumu (gymnasium.check_env).

torch/SB3 gerektirmeden Gym arayüzünü doğrular: reset/step imzaları, obs-space
uyumu, 5'li dönüş, truncated bayrağı. SB3 eğitimi kullanıcının bir sonraki adımı.
"""
import numpy as np


def test_check_env():
    from gymnasium.utils.env_checker import check_env
    from tma.gym_env import make_tma_gym
    env = make_tma_gym(seed=0)
    check_env(env, skip_render_check=True)


def test_rollout_and_truncation():
    from tma.gym_env import make_tma_gym
    env = make_tma_gym(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32
    steps, truncated = 0, False
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert terminated is False           # asla terminal (yalnızca truncation)
        if truncated:
            break
        assert steps < 100
    assert truncated is True and "pos_err" in info


if __name__ == "__main__":
    test_check_env()
    test_rollout_and_truncation()
    print("TMAGym: check_env geçti; rollout truncated=True ile bitti, terminated hep False. GEÇTI.")
