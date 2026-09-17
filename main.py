"""main.py — scenarios.json'dan RL eğitimi (Faz 3, SB3 PPO/DQN).

TMAGym (tma/gym_env.py) + FixedScenario (tma/initializer.py) kullanır: her
reset AYNI geometri (scenarios.json), yalnızca ölçüm gürültüsü rastgele —
"Durum 2 / kıyas" kurulumu, tek bir senaryoya karşı ajan eğitmek/ayıklamak
için. Genel eğitim (domain randomization, "Durum 1") istersen --randomized
bayrağıyla RandomizedScenario'ya geç.

Kullanım — IDE'den: aşağıdaki AYARLAR bloğunu düzenleyip Run'a basın.
           Terminalden: aynı ayarlar --bayrak olarak da geçilebilir, örn.
    python main.py --timesteps 500000 --out models/dqn_fixed --algo dqn
    python main.py --randomized --timesteps 1000000        # domain randomization ile eğitim

Not: SB3 + torch gerekir (pip install stable-baselines3 torch). Windows/Python
3.13'te torch c10.dll sorunu için 3.12 venv önerilir (bkz. README.md / DEVAM.md).
"""
import argparse
import os
import time

import numpy as np

from tma import runconfig
from tma.gym_env import make_tma_gym
from tma.initializer import FixedScenario

# ==================== AYARLAR (IDE'den çalıştırırken burayı düzenleyin) ====================
OUT = "models/tma_model"       # eğitilen modelin kaydedileceği yol (.zip uzantısız)
ALGO = "ppo"                   # "ppo" | "dqn" (SB3) | "npdqn" (tma/dqn.py, saf numpy)
SCENARIO = "scenarios.json"    # FixedScenario JSON yolu
RANDOMIZED = False             # True: scenarios.json yerine domain randomization
OBS = "los"                    # "los" | "absolute"
TURN_RATE_DEG = 3.0
OPENING_LEG = True             # açılış bacağı koşulsun mu
OPENING_OFFSET_DEG = 90.0      # açılış bacağının kerterize ofseti:
                               #   90 = kerterize dik (klasik optimal açılış)
                               #    0 = ilk temas kerterizine git (zayıf-gözlenebilirlik tabanı)
                               # DEĞERLENDIRMEDE de aynısını kullan (analyze.py)
TOTAL_TIME = 1200.0            # episode uzunluğu (s); None ise world.py'nin T_TOTAL'i

TIMESTEPS = 200_000            # eğitim adım sayısı
SEED = 0
# ==========================================================================================


def build_env(args):
    initializer = None if args.randomized else FixedScenario.from_json(args.scenario)
    return make_tma_gym(seed=args.seed, obs=args.obs,
                        opening_leg=args.opening_leg,
                        opening_offset_deg=args.opening_offset_deg,
                        turn_rate_deg=args.turn_rate_deg, initializer=initializer,
                        total_time=args.total_time)


def train_numpy_dqn(env, timesteps, seed=0, eps0=1.0, eps_min=0.05,
                    explore_frac=0.1, log_every=10_000):
    """tma/dqn.py'nin saf-numpy DQN'i, SB3'süz, aynı gym ortamında.

    Bellman hedefindeki done = `terminated`. TMAGym süre dolunca
    truncated=True / terminated=False döndürür (bkz. tma/gym_env.py başlığı):
    zaman-limiti kesilmesi görev bitimi DEĞILDIR, o yüzden orada bootstrap SÜRER.
    Epsilon SB3 DQN'le aynı biçimde (ilk %10'da lineer 1.0 -> 0.05) düşürülür.
    """
    from tma.dqn import DQN as NpDQN
    agent = NpDQN(obs_dim=env.observation_space.shape[0],
                  n_act=int(env.action_space.n), seed=seed)
    obs, _ = env.reset(seed=seed)
    n_explore = max(1, int(explore_frac * timesteps))
    ep_r, ep_rs, t0 = 0.0, [], time.perf_counter()
    for t in range(timesteps):
        eps = max(eps_min, eps0 + (eps_min - eps0) * t / n_explore)
        a = agent.act(np.asarray(obs, float), eps)
        obs2, r, terminated, truncated, _ = env.step(a)
        agent.buf.add(obs, a, r, obs2, terminated)
        agent.train_step()
        obs, ep_r = obs2, ep_r + r
        if terminated or truncated:
            ep_rs.append(ep_r); ep_r = 0.0
            obs, _ = env.reset()
        if (t + 1) % log_every == 0:
            m = np.mean(ep_rs) if ep_rs else float("nan")
            print(f"  adim {t+1:7d} | eps {eps:.3f} | {len(ep_rs)} episode ort odul {m:8.3f}",
                  flush=True)
            ep_rs = []
    print(f"  egitim suresi: {time.perf_counter() - t0:.0f} s")
    return agent


def main():
    # Varsayılanlar yukarıdaki AYARLAR bloğundan gelir; bayraklar sadece üzerine yazar.
    p = argparse.ArgumentParser(description="TMA RL eğitimi (SB3 PPO/DQN)")
    p.add_argument("--scenario", default=SCENARIO, help="FixedScenario JSON yolu")
    p.add_argument("--randomized", action="store_true",
                   help="scenarios.json yerine domain randomization (RandomizedScenario)")
    p.add_argument("--algo", choices=["ppo", "dqn", "npdqn"], default=ALGO,
                   help="ppo/dqn = SB3; npdqn = tma/dqn.py (saf numpy, torch gerekmez)")
    p.add_argument("--obs", choices=["absolute", "los"], default=OBS)
    p.add_argument("--turn-rate-deg", type=float, default=TURN_RATE_DEG)
    p.add_argument("--no-opening-leg", action="store_true")
    p.add_argument("--opening-offset-deg", type=float, default=OPENING_OFFSET_DEG,
                   help="açılış bacağının kerterize ofseti: 90 dik (varsayılan), "
                        "0 ilk temas kerterizine git")
    p.add_argument("--total-time", type=float, default=TOTAL_TIME,
                   help="episode uzunluğu (s); None ise world.py'nin T_TOTAL'i kullanılır")
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    args.randomized = RANDOMIZED or args.randomized
    args.opening_leg = OPENING_LEG and not args.no_opening_leg

    env = build_env(args)
    print(f"[main] senaryo={'randomized' if args.randomized else args.scenario} | "
          f"algo={args.algo} | obs={args.obs} | turn_rate_deg={args.turn_rate_deg} | "
          f"opening_offset_deg={args.opening_offset_deg} | "
          f"total_time={args.total_time} | timesteps={args.timesteps} | out={args.out}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)      # models/ yoksa oluştur

    if args.algo == "npdqn":
        agent = train_numpy_dqn(env, args.timesteps, seed=args.seed)
        agent.save(args.out + ".npz")
        print(f"[main] kaydedildi: {args.out}.npz")
    else:
        try:
            from stable_baselines3 import PPO, DQN
        except ImportError:
            raise SystemExit(
                "stable-baselines3/torch bulunamadı. Kurulum: pip install stable-baselines3 torch\n"
                "(Windows/Python 3.13'te torch c10.dll sorunu için 3.12 venv önerilir.)\n"
                "SB3 istemiyorsanız --algo npdqn saf numpy ile eğitir."
            )
        Model = {"ppo": PPO, "dqn": DQN}[args.algo]
        extra = {"n_steps": 2048} if args.algo == "ppo" else {}
        model = Model("MlpPolicy", env, seed=args.seed, verbose=1, **extra)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.out)
        print(f"[main] kaydedildi: {args.out}.zip")

    # Eğitim koşullarını modelin yanına yaz — analyze.py bunu okuyup değerlendirme
    # ayarıyla farkını ("KAYMA") bastırır. Bkz. tma/runconfig.py.
    cfg_path = runconfig.save(args.out, {
        "scenario": args.scenario, "randomized": args.randomized, "obs": args.obs,
        "opening_leg": args.opening_leg, "opening_offset_deg": args.opening_offset_deg,
        "turn_rate_deg": args.turn_rate_deg, "total_time": args.total_time,
        "algo": args.algo, "timesteps": args.timesteps, "seed": args.seed,
    })
    print(f"[main] koşum ayarı:  {cfg_path}")


if __name__ == "__main__":
    main()
