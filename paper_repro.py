"""Ristic & Arulampalam (2026) reprodüksiyonu — bizim pluggable mimaride, SB3'süz DQN.

build_paper_world: CKF + PaperInitializer + Paper12Obs + ParetoTerminalReward + PTB
açılış bacağı + 16 aksiyon + tek-karar episode (leg1=PTB reset içinde, leg2=tek karar).
train_dqn / evaluate: sıfırdan numpy DQN ile eğitim; DQN/PTB/ITO'nun MC kıyası.

Kullanım — IDE'den: aşağıdaki AYARLAR bloğunu düzenleyip Run'a basın.
           Terminalden: pozisyonel biçim de çalışır (AYARLAR'ın üzerine yazar):
    python paper_repro.py train 0.7 50000 models/agent_b07.npz
    python paper_repro.py eval 5000 0.7,models/agent_b07.npz 0.9,models/agent_b09.npz

Yalnızca numpy gerekir — SB3/torch yok.
"""
import os
import time
import numpy as np

from tma.world import World, Platform
from tma.sensor import BearingOnlySensor
from tma.filter_ckf import CubatureKF
from tma.obs import Paper12Obs
from tma.reward import ParetoTerminalReward
from tma.initializer import PaperInitializer
from tma.maneuver import PaperPTBManeuver, ITOManeuver, ptb_heading
from tma.dqn import DQN

# --- makale parametreleri (Tablo II) ---
M = 12                       # bacak başına adım
DT = 1.0
SPEED = 1.0
SIGMA_RAD = 0.0175           # ~1 derece
Q_INT = 1e-6
R0 = 23.0
SIGMA_R = 5.0
VMAX = 3.0
NA = 16
ACTIONS = np.deg2rad(np.arange(0, 360, 360 / NA))     # 22.5 derece aralık

# ==================== AYARLAR (IDE'den çalıştırırken burayı düzenleyin) ====================
MODE = "eval"                  # "eval" (kıyas tablosu) | "train" (ajan eğit)
MODELS = "models"              # eğitilmiş ajan .npz dosyalarının klasörü

# --- train modu ---
TRAIN_BETA = 0.7               # Pareto ödül ağırlığı: r = -dE^beta * dM^(1-beta)
TRAIN_EPISODES = 30_000        # makale ölçeği 50k (~2.5 dk / 30k episode)
TRAIN_OUT = None               # None ise MODELS/agent_b0{beta*10}.npz olarak türetilir

# --- eval modu ---
EVAL_EPISODES = 3000           # Monte Carlo episode sayısı (makale ölçeği 5000)
EVAL_BETAS = [0.7]             # değerlendirilecek ajanlar; dosya adı beta'dan türetilir.
                               # Beşini birden kıyaslamak için: [0.1, 0.3, 0.5, 0.7, 0.9]
EVAL_BASELINES = True          # PTB ve ITO taban çizgileri de koşulsun mu
# ==========================================================================================


def agent_path(beta, models_dir=None):
    """beta -> eğitilmiş ajan dosyası. 0.7 -> models/agent_b07.npz (paper_figs.py ile aynı)."""
    return f"{models_dir or MODELS}/agent_b0{int(round(beta * 10))}.npz"


def _opening_ptb(world):
    return ptb_heading(world.ownship.filter.estimate, world.ownship.pos,
                       world.ownship.speed, world.dt, M * DT)


def build_paper_world(beta, maneuver=None, seed=0):
    sensor = BearingOnlySensor(sigma_deg=np.rad2deg(SIGMA_RAD))
    filt = CubatureKF(sensor, dt=DT, q=Q_INT, R0=R0, sigma_R=SIGMA_R, vmax=VMAX)
    ownship = Platform(pos=np.zeros(2), hdg=0.0, speed=SPEED, turn_rate_max=np.inf,
                       sensor=sensor, filter=filt, maneuver=maneuver)
    target = Platform(pos=np.zeros(2), vel=np.zeros(2))
    return World(ownship, target, reward_fn=ParetoTerminalReward(beta),
                 initializer=PaperInitializer(speed=SPEED), obs_encoder=Paper12Obs(),
                 seed=seed, dt=DT, total_time=2 * M * DT, plan_every=M,
                 actions=ACTIONS, opening_leg=True, opening_heading_fn=_opening_ptb)


def _metrics(world):
    true = world.true_target_state()
    diff = world.ownship.filter.estimate[:2] - true[:2]
    dE = float(np.linalg.norm(diff))
    Ppos = world.ownship.filter.covariance[:2, :2]
    dM = float(np.sqrt(max(diff @ np.linalg.solve(Ppos + 1e-12*np.eye(2), diff), 0.0)))
    return dE, dM


def train_dqn(beta=0.7, episodes=30000, eps0=1.0, eps_min=0.05, eps_decay=0.99994,
              seed=0, log_every=5000):
    agent = DQN(obs_dim=12, n_act=NA, seed=seed)
    world = build_paper_world(beta, maneuver=None, seed=10_000 + seed)
    eps = eps0
    t0 = time.perf_counter()
    recent = []
    for ep in range(episodes):
        obs = world.reset()                       # leg1 (PTB) reset içinde koşar
        a = agent.act(obs, eps)
        _, r, done, _ = world.apply(a)            # leg2 = tek karar, terminal
        agent.buf.add(obs, a, r)
        agent.train_step()
        eps = max(eps_min, eps * eps_decay)
        recent.append(r)
        if (ep + 1) % log_every == 0:
            print(f"  ep {ep+1:6d} | eps {eps:.3f} | son {log_every} ort ödül "
                  f"{np.mean(recent):7.3f}", flush=True)
            recent = []
    print(f"  eğitim süresi: {time.perf_counter()-t0:.0f} s ({episodes} episode)")
    return agent


def evaluate_dqn(agent, beta, n_ep=3000, seed0=100_000):
    dE, dM = [], []
    for e in range(n_ep):
        world = build_paper_world(beta, maneuver=None, seed=seed0 + e)
        obs = world.reset()
        world.apply(agent.greedy(obs))
        e_, m_ = _metrics(world)
        dE.append(e_); dM.append(m_)
    return np.array(dE), np.array(dM)


def evaluate_baseline(make_maneuver, beta, n_ep=3000, seed0=100_000):
    dE, dM = [], []
    for e in range(n_ep):
        man = make_maneuver()
        world = build_paper_world(beta, maneuver=man, seed=seed0 + e)
        world.reset()
        world.apply(man.decide(world.maneuver_context()))
        e_, m_ = _metrics(world)
        dE.append(e_); dM.append(m_)
    return np.array(dE), np.array(dM)


def _row(name, dE, dM):
    print(f"{name:12s} | {dE.mean():7.3f} | {dE.std():7.3f} | {dE.max():8.2f} | {dM.mean():6.3f}")


if __name__ == "__main__":
    import sys

    # Argüman verilmezse yukarıdaki AYARLAR bloğu kullanılır (IDE'den Run'a basmak için).
    # Pozisyonel argümanlar verilirse onlar kazanır — eski komutlar aynen çalışır.
    argv = sys.argv[1:]
    mode = argv[0] if argv else MODE

    if mode == "train":
        beta = float(argv[1]) if len(argv) > 1 else TRAIN_BETA
        episodes = int(argv[2]) if len(argv) > 2 else TRAIN_EPISODES
        out = argv[3] if len(argv) > 3 else (TRAIN_OUT or agent_path(beta))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        print(f"[beta={beta}] {episodes} episode eğitiliyor -> {out}")
        agent = train_dqn(beta=beta, episodes=episodes, seed=0)
        agent.save(out)
        print("kaydedildi:", out)

    elif mode == "eval":
        n_eval = int(argv[1]) if len(argv) > 1 else EVAL_EPISODES
        if len(argv) > 2:                       # "beta,dosya" çiftleri
            specs = [(float(s.split(",")[0]), s.split(",", 1)[1]) for s in argv[2:]]
        else:                                   # AYARLAR: dosya adı beta'dan türetilir
            specs = [(b, agent_path(b)) for b in EVAL_BETAS]

        found = [(b, fn) for b, fn in specs if os.path.exists(fn)]
        for b, fn in specs:
            if not os.path.exists(fn):
                print(f"[atlandı] eğitilmiş ajan yok: {fn}\n"
                      f"          üretmek için: python paper_repro.py train {b} 50000 {fn}")
        if not found and not EVAL_BASELINES:
            raise SystemExit("Değerlendirilecek bir şey kalmadı.")

        print(f"\n{n_eval} episode\n")
        print(f"{'Yontem':12s} | {'ort dE':>7s} | {'std dE':>7s} | {'maks dE':>8s} | {'ort dM':>6s}")
        print("-" * 52)
        for b, fn in found:
            agent = DQN(obs_dim=12, n_act=NA, seed=0); agent.load(fn)
            dE, dM = evaluate_dqn(agent, b, n_ep=n_eval)
            _row(f"DQN b={b}", dE, dM)
        if EVAL_BASELINES:
            # beta yalnızca ödülü etkiler; PTB/ITO ödülü kullanmadığı için değeri önemsiz
            dE, dM = evaluate_baseline(PaperPTBManeuver, 0.5, n_ep=n_eval); _row("PTB", dE, dM)
            dE, dM = evaluate_baseline(ITOManeuver, 0.5, n_ep=n_eval); _row("ITO", dE, dM)

    else:
        raise SystemExit(f"Bilinmeyen mod: {mode!r}. Beklenen: 'train' veya 'eval'.\n"
                         "  python paper_repro.py train 0.7 50000 models/agent_b07.npz\n"
                         "  python paper_repro.py eval 5000 0.7,models/agent_b07.npz\n"
                         "  (argümansız: dosyanın başındaki AYARLAR bloğu kullanılır)")
