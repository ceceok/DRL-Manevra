"""TMAGym — ince Gymnasium adaptörü.

TMA mantığı yok; yalnızca World'ün reset/apply'ını Gymnasium arayüzüne çevirir.
gymnasium yalnızca burada (tembel import) — çekirdek saf numpy kalır.

RL eğitim ortamı olarak tam tasarımı kullanır: obs='los' (Faz 3), opening_leg=True +
turn_rate_deg (Faz 2), RandomizedScenario (eğitim). maneuver=None: RL ajanı dışarıda
(SB3); World açılış bacağını yine koşar (privileged değil).

opening_offset_deg açılış bacağının kerterize göre ofseti: 90 kerterize dik (varsayılan),
0 ilk temas kerterizine git. Ajan episode'u bu bacaktan SONRA gördüğü için ofset ajanın
başlangıç durumunu değiştirir — eğitim ve değerlendirmede AYNI değeri kullan.

terminated=False, truncated=done: episode 'çözüldü' ile değil, süre dolduğu için
biter -> her zaman truncation. SB3 bootstrap'ı bu sayede doğru (GAE truncation).
"""
import numpy as np


def make_tma_gym(seed=0, obs="los", opening_leg=True, turn_rate_deg=3.0,
                 initializer=None, total_time=None, opening_offset_deg=90.0):
    import gymnasium as gym
    from gymnasium import spaces
    from .registry import build_world

    class TMAGym(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.world = build_world(seed=seed, maneuver=None, obs=obs,
                                     opening_leg=opening_leg, turn_rate_deg=turn_rate_deg,
                                     opening_offset_deg=opening_offset_deg,
                                     initializer=initializer, total_time=total_time)
            obs0 = self.world.reset(seed=seed)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs0.shape[0],), dtype=np.float32)
            self.action_space = spaces.Discrete(len(self.world.actions))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            o = self.world.reset(seed=seed)
            return o.astype(np.float32), {}

        def step(self, action):
            o, r, done, info = self.world.apply(int(action))
            return o.astype(np.float32), float(r), False, bool(done), info

    return TMAGym()
