"""Makale figürleri (Fig 1, Fig 2) ve Table III üretimi.

python paper_figs.py     ->  graphics/fig1_scenario.png, fig2_methods.png,
                             table3.png, table3.csv (+ konsola tablo)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from paper_repro import (build_paper_world, evaluate_dqn, evaluate_baseline,
                         _metrics, ACTIONS, M, DT)
from tma.dqn import DQN
from tma.maneuver import PaperPTBManeuver, ITOManeuver
from tma.reward import ParetoTerminalReward

OUT = "graphics"
MODELS = "models"
NA = len(ACTIONS)


def cov_ellipse(ax, mean, P2, nsig=2.0, **kw):
    vals, vecs = np.linalg.eigh(P2)
    vals = np.maximum(vals, 1e-9)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nsig * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, **kw))


def _run_episode(make_maneuver, beta, seed, agent=None):
    """Tam episode koşar (leg1 PTB reset içinde + leg2 karar); iz + son belief döndürür."""
    if agent is not None:
        w = build_paper_world(beta, maneuver=None, seed=seed)
        obs = w.reset()
        a = agent.greedy(obs)
    else:
        man = make_maneuver()
        w = build_paper_world(beta, maneuver=man, seed=seed)
        w.reset()
        a = man.decide(w.maneuver_context())
    own_before = np.asarray(w.ownship.hist)          # leg1 izi (karar öncesi)
    n_leg1 = len(own_before) - 1
    w.apply(a)
    own = np.asarray(w.ownship.hist)                 # tüm iz (leg1 + leg2)
    tgt = np.array([w.target.pos_at(t) for t in DT * np.arange(len(own))])
    est = w.ownship.filter.estimate
    P = w.ownship.filter.covariance
    dE, dM = _metrics(w)
    return dict(own=own, n_leg1=n_leg1, tgt=tgt, est=est, P=P, dE=dE, dM=dM, action=a)


# ---------------- Fig 1: illustrative scenario ----------------
def fig1(seed=7):
    w = build_paper_world(0.7, maneuver=None, seed=seed)
    w.reset()                                        # leg1 (PTB) koşuldu
    own = np.asarray(w.ownship.hist)
    dec = own[-1]                                    # karar anındaki gözlemci konumu
    tgt_now = w.target.pos_at(w.t)
    tgt_half = np.array([w.target.pos_at(t) for t in DT * np.arange(M + 1)])
    est = w.ownship.filter.estimate
    P = w.ownship.filter.covariance

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    L = 26
    for j, th in enumerate(ACTIONS):                 # 16 aday leg2 yönü
        v = np.array([np.sin(th), np.cos(th)])
        ax.plot([dec[0], dec[0] + L * v[0]], [dec[1], dec[1] + L * v[1]],
                '--', color="tab:blue", lw=0.8, alpha=0.55, zorder=1)
        ax.annotate(str(j + 1), dec + (L + 2) * v, fontsize=6.5, color="#3b5", ha="center")
    ax.plot(own[:, 0], own[:, 1], '-', color="tab:blue", lw=2.2, label="Gözlemci (leg 1)")
    ax.plot(*dec, 's', color="tab:blue", ms=9, label="Gözlemci (karar anı)")
    ax.plot(tgt_half[:, 0], tgt_half[:, 1], '-', color="gray", lw=1.6, label="Hedef izi (ilk yarı)")
    ax.plot(*tgt_now, 's', color="k", ms=8, label="Hedef (karar anı)")
    ax.plot(*est[:2], '^', color="tab:red", ms=7, label="CKF kestirim")
    cov_ellipse(ax, est[:2], P[:2, :2], nsig=2.0, ec="tab:red", lw=1.6)
    ax.set_xlabel("X (a.u.)"); ax.set_ylabel("Y (a.u.)")
    ax.set_title("Fig 1 — Senaryo: 16 aday leg-2 yönü + CKF posterior (2σ)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_scenario.png", dpi=140); plt.close(fig)
    print("kaydedildi: fig1_scenario.png")


# ---------------- Fig 2: three methods, same geometry ----------------
def fig2(seed=123, agent07=None):
    runs = [
        ("DQN (β=0.7)", _run_episode(None, 0.7, seed, agent=agent07), "tab:green"),
        ("PTB", _run_episode(PaperPTBManeuver, 0.5, seed), "tab:orange"),
        ("ITO", _run_episode(ITOManeuver, 0.5, seed), "tab:purple"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (name, r, col) in zip(axes, runs):
        own, n1 = r["own"], r["n_leg1"]
        ax.plot(own[:n1 + 1, 0], own[:n1 + 1, 1], '-', color="tab:blue", lw=2.0, label="leg 1")
        ax.plot(own[n1:, 0], own[n1:, 1], '-', color=col, lw=2.2, label="leg 2")
        ax.plot(*own[0], 's', color="green", ms=8, label="Gözlemci başlangıç")
        ax.plot(*own[-1], 's', color="k", ms=7, label="Gözlemci son")
        ax.plot(r["tgt"][:, 0], r["tgt"][:, 1], '-', color="k", lw=1.4, label="Hedef izi")
        ax.plot(*r["tgt"][0], 'o', mfc="none", mec="k", ms=8)
        ax.plot(*r["tgt"][-1], '*', color="k", ms=13, label="Hedef (son)")
        ax.plot(*r["est"][:2], '^', color="tab:red", ms=8, label="CKF kestirim")
        cov_ellipse(ax, r["est"][:2], r["P"][:2, :2], nsig=2.0, ec="tab:red", lw=1.5)
        ax.set_title(f"{name} | aksiyon {r['action']+1} | dE={r['dE']:.2f} a.u.")
        ax.set_xlabel("X (a.u.)"); ax.set_ylabel("Y (a.u.)")
        ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Fig 2 — Aynı geometride üç yöntem (tek episode)", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_methods.png", dpi=140); plt.close(fig)
    print("kaydedildi: fig2_methods.png")


# ---------------- Table III ----------------
def _reward_mean(dE, dM, beta):
    return np.mean(-(dE**beta) * (dM**(1.0 - beta)))


def table3(n_eval=3000):
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    cols, results = [], {}
    for b in betas:
        agent = DQN(obs_dim=12, n_act=NA, seed=0)
        agent.load(f"{MODELS}/agent_b0{int(b*10)}.npz")
        dE, dM = evaluate_dqn(agent, b, n_ep=n_eval)
        results[f"β={b}"] = (dE, dM, _reward_mean(dE, dM, b)); cols.append(f"β={b}")
    for name, mk in [("PTB", PaperPTBManeuver), ("ITO", ITOManeuver)]:
        dE, dM = evaluate_baseline(mk, 0.5, n_ep=n_eval)
        results[name] = (dE, dM, _reward_mean(dE, dM, 0.5)); cols.append(name)

    rows = ["Ort dE", "Std dE", "Maks dE", "Ort dM", "Ort ödül†"]
    table = np.zeros((len(rows), len(cols)))
    for j, c in enumerate(cols):
        dE, dM, rw = results[c]
        table[:, j] = [dE.mean(), dE.std(), dE.max(), dM.mean(), rw]

    # konsol
    print("\nTABLE III — " + f"MC {n_eval} episode")
    print(f"{'Metrik':10s} | " + " | ".join(f"{c:>8s}" for c in cols))
    print("-" * (13 + 11 * len(cols)))
    for i, rname in enumerate(rows):
        best = np.argmin(table[i]) if i < 4 else np.argmax(table[i])
        cells = []
        for j in range(len(cols)):
            s = f"{table[i, j]:8.3f}"
            cells.append(f"*{s.strip()}*".rjust(8) if j == best else s)
        print(f"{rname:10s} | " + " | ".join(cells))
    print("† DQN kendi β'sını; PTB/ITO β=0.5 referansını kullanır.")

    # csv
    import csv
    with open(f"{OUT}/table3.csv", "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["Metrik"] + cols)
        for i, rname in enumerate(rows):
            wtr.writerow([rname] + [f"{table[i, j]:.3f}" for j in range(len(cols))])

    # png tablo
    fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off")
    cell = [[f"{table[i, j]:.3f}" for j in range(len(cols))] for i in range(len(rows))]
    tbl = ax.table(cellText=cell, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for i in range(len(rows)):
        best = np.argmin(table[i]) if i < 4 else np.argmax(table[i])
        tbl[(i + 1, best)].set_facecolor("#d7ecd9")
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#eef"); tbl[(0, j)].set_text_props(fontweight="bold")
    ax.set_title("TABLE III — Performans karşılaştırması (Monte Carlo)", fontweight="bold", pad=14)
    plt.savefig(f"{OUT}/table3.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("kaydedildi: table3.png, table3.csv")


if __name__ == "__main__":
    import sys
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    os.makedirs(OUT, exist_ok=True)
    a07 = DQN(obs_dim=12, n_act=NA, seed=0); a07.load(f"{MODELS}/agent_b07.npz")
    fig1(seed=7)
    fig2(seed=123, agent07=a07)
    table3(n_eval=n_eval)
