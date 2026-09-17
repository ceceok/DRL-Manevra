"""analyze.py — RL ajanı (SB3 PPO/DQN ya da tma/dqn.py) ile klasik planlayıcıların kıyası.

İki analiz tek dosyada (ikisi de tma/rollout.py'deki aynı koşum motorunu kullanır):
  1) Monte Carlo : N episode, episode sonu kestirim hatası istatistikleri
                   (pos_err = |filtre kestirimi - gerçek hedef|, metre)
  2) Tek koşum   : karar tablosu (t, kerteriz, kendi rota, seçilen aksiyon)
                   + XY iz / rota-zaman grafiği (interaktif pencere)

Kullanım — IDE'den: aşağıdaki AYARLAR bloğunu düzenleyip Run'a basın.
           Terminalden: aynı ayarlar --bayrak olarak da geçilebilir, örn.
    python analyze.py --episodes 500       # daha büyük MC
    python analyze.py --episodes 0         # sadece grafik
    python analyze.py --no-plot            # sadece istatistik
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from tma import runconfig
from tma.rollout import run_agent_episode, run_baseline_episode
from tma.maneuver import PTBManeuver, _fim_logdet   # FIM'in tek kaynağı: planlayıcılarla aynı fonksiyon

# ==================== AYARLAR (IDE'den çalıştırırken burayı düzenleyin) ====================
# --- KIYASLANACAK AJANLAR: (görünen ad, model yolu (uzantısız), algo)
# algo None ise modelin sidecar'ından okunur. Kaç ajan yazarsanız hepsi AYNI koşuda,
# aynı tohumlarda, klasik planlayıcılarla yan yana kıyaslanır. Liste boşsa yalnızca
# klasik planlayıcılar koşar.
#
# Değerlendirme DÜNYASI hepsinde ortaktır (aksi hâlde kıyas anlamsız olur); referans
# ilk ajanın eğitim ayarıdır, üzerine OVERRIDES geçer. Ama KAYMA ajan başına
# raporlanır — farklı koşullarda eğitilmiş ajanları aynı dünyada yarıştırıp hangisinin
# daha iyi genellediğini görebilirsiniz. `obs` ajanın kendi eğitim değerinden gelir
# (gözlem kodlaması ajanın arayüzüdür, dünyanın özelliği değil).
AGENTS = [
    ("PPO t1200", "models/tma_model_t1200", None),
    # ("npDQN", "models/npdqn_los", None),
]

# --- eğitim koşulları: modelin yanındaki sidecar'dan okunur (models/x.run.json).
# Sidecar yoksa (main.py'nin bu sürümünden ÖNCE eğitilmiş modeller) aşağıdaki
# varsayım kullanılır ve uyarı basılır. Yeniden eğitince sidecar kendiliğinden oluşur.
FALLBACK_TRAIN = {"scenario": "scenarios.json", "randomized": False, "obs": "los",
                  "opening_leg": True, "opening_offset_deg": 90.0,
                  "turn_rate_deg": 3.0, "total_time": 1200.0, "algo": "ppo"}

# --- KASITLI SAPMALAR: "ajan eğitilmediği koşulda nasıl davranıyor?" deneyi.
# Buraya yazdığınız her anahtar eğitim ayarının üzerine geçer ve çıktıda KAYMA
# olarak bastırılır. Boş bırakırsanız ajan eğitim koşullarında değerlendirilir.
# Klasik planlayıcılar (RH/Kahin/PTB) kaymadan etkilenmez — kontrol grubu onlar.
#   opening_offset_deg: 90 = kerterize dik (klasik optimal açılış)
#                        0 = ilk temas kerterizine git (zayıf-gözlenebilirlik tabanı)
# Tek düğmeyi tarayıp genelleme eğrisi çıkarmak için diğerlerini yorum satırı yapın.
OVERRIDES = {
    "turn_rate_deg": 5.0,
    "opening_offset_deg": 10.0,
    "total_time": 1500.0,
    "randomized": False,       # analyze her zaman FixedScenario koşar (bkz. tma/rollout.py)
}

ALGO = "ppo"                     # sidecar da CLI de vermezse son çare

EPISODES = 0                     # Monte Carlo episode sayısı — 0 ise MC atlanır
SEED0 = 500_000                    # MC ilk tohumu
PLOT = True                        # tek koşum grafiği çizilsin mi
PLOT_SEED = None                   # görselleştirilecek koşum; None ise SEED0
SHOW_DECISION_STATS = True         # karar anı bazında rota önerisi dağılımı tablosu
SHOW_TIME_SERIES = True            # MC zaman serisi figürü: RMSE(t), dM(t), NEES(t)

# karar anlarındaki kerteriz çizgileri (XY panelinde)
SHOW_MEAS_RAYS = False              # ham sensör ölçümü (noktalı, yöntem renginde)
SHOW_TRUTH_RAYS = True             # gerçek (ground truth) kerteriz (ince düz, gri)

# PTB varyantları: kerterize göre ofset açıları. Listeye kaç açı yazarsanız o kadar
# PTB aynı koşuda yan yana kıyaslanır. Örn. [90] tek varyant, [90, 80, 70] üç varyant.
PTB_OFFSETS_DEG = [90.0,70.0,50.0,30]
# ==========================================================================================

# (görünen ad, koşum anahtarı) — anahtar str ise registry, çağrılabilir ise parametreli
# manevra fabrikası, AgentRun ise eğitilmiş ajan. PTB varyantları aşağıda, ajanlar
# main() içinde (model yüklendikten sonra) otomatik eklenir.
METHODS = [("RH", "rh"), ("Kahin FIM", "oracle")]
PLOT_SKIP = ("Rastgele",)      # istatistikte faydalı, grafikte gürültü

COLORS = {"RH": "tab:orange", "Rastgele": "tab:gray", "Kahin FIM": "tab:red"}
# çizgiler üst üste bindiğinde renk tek başına ayırt etmeye yetmiyor
LINESTYLES = {"RH": "--", "Rastgele": ":", "Kahin FIM": (0, (3, 1, 1, 1))}
MARKERS = {"RH": "s", "Rastgele": "x", "Kahin FIM": "^"}


class AgentRun:
    """Eğitilmiş bir ajan + onun gözlem kodlaması. METHODS'ta koşum anahtarı olarak durur.

    obs ajanla birlikte taşınır çünkü gözlem kodlaması ajanın ARAYÜZÜ'dür; dünyanın
    özelliği değil. Böylece 'los' ile eğitilmiş bir ajanla 'absolute' ile eğitilmiş
    başka bir ajan aynı dünyada yarışabilir.
    """

    def __init__(self, model, obs):
        self.model, self.obs = model, obs


# --- PTB varyantlarını kaydet (ad + stil otomatik) ---
_PTB_COLORS = ["tab:purple", "magenta", "darkviolet", "orchid", "indigo"]
_PTB_LINESTYLES = ["-.", (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1)), (0, (2, 2))]
_PTB_MARKERS = ["D", "v", "P", "*", "h"]

for _i, _off in enumerate(PTB_OFFSETS_DEG):
    _name = f"PTB {_off:g}°"
    METHODS.append((_name, (lambda o: (lambda: PTBManeuver(offset_deg=o)))(_off)))
    COLORS[_name] = _PTB_COLORS[_i % len(_PTB_COLORS)]
    LINESTYLES[_name] = _PTB_LINESTYLES[_i % len(_PTB_LINESTYLES)]
    MARKERS[_name] = _PTB_MARKERS[_i % len(_PTB_MARKERS)]

# --- ajan stilleri (main() içinde adlarıyla eşleştirilir) ---
_AG_COLORS = ["tab:green", "tab:blue", "tab:cyan", "tab:olive", "seagreen"]
_AG_LINESTYLES = ["-", (0, (6, 2)), (0, (3, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (2, 1))]
_AG_MARKERS = ["o", "^", "s", "d", "X"]

METHOD_ORDER = {}          # tablo sıralaması — ajanlar eklendikten sonra main()'de kurulur


# ---------------- koşum ----------------
def run_one(key, args, seed):
    """Bir episode koşar; (world, info) döner. key AgentRun ise eğitilmiş ajan."""
    if isinstance(key, AgentRun):
        return run_agent_episode(key.model, args.scenario, key.obs, args.opening_leg,
                                 args.turn_rate_deg, seed, total_time=args.total_time,
                                 opening_offset_deg=args.opening_offset_deg)
    return run_baseline_episode(key, args.scenario, args.opening_leg,
                                args.turn_rate_deg, seed, total_time=args.total_time,
                                opening_offset_deg=args.opening_offset_deg)


# ---------------- 1) Monte Carlo ----------------
def _circ_stats(deg):
    """Açılar için dairesel ortalama ve std (derece).

    Düz ortalama açılarda yanlıştır (350° ile 10°'un ortalaması 180° çıkar, doğrusu 0°).
    Dairesel std = sqrt(-2 ln R); R = bileşke vektör uzunluğu (R→1 dar, R→0 dağınık).
    """
    a = np.radians(np.asarray(deg, float))
    C, S = np.cos(a).mean(), np.sin(a).mean()
    R = min(np.hypot(C, S), 1.0)
    mean = np.degrees(np.arctan2(S, C)) % 360.0
    var = -2.0 * np.log(max(R, 1e-12))
    std = np.degrees(np.sqrt(var)) if var > 0.0 else 0.0    # R=1 -> tam 0 (-0.0 değil)
    return mean, std


def _mahalanobis(d, P):
    """sqrt(dᵀ P⁻¹ d), adım adım (d: (K,2), P: (K,2,2))."""
    q = np.einsum("ki,kij,kj->k", d, np.linalg.inv(P + 1e-12 * np.eye(2)), d)
    return np.sqrt(np.maximum(q, 0.0))


def _error_series(world):
    """Her dt anında konum ve hız için (hata, Mahalanobis) — Ristic'in metrikleri.

    dE = |kestirim - gerçek|                    → doğruluk (konum: m, hız: m/s)
    dM = sqrt(dᵀ P⁻¹ d), birimsiz               → tutarlılık (filtre kovaryansına göre)
    """
    est = np.asarray(world.est_hist)                      # (K,4)
    P = np.asarray(world.P_hist)                          # (K,4,4)
    t = world.dt * np.arange(len(est))
    tgt = world.target.pos0 + np.outer(t, world.target.vel)
    dp = est[:, :2] - tgt                                 # konum hatası
    dv = est[:, 2:] - world.target.vel                    # hız hatası (hedef CV, sabit hız)
    return (t, np.linalg.norm(dp, axis=1), _mahalanobis(dp, P[:, :2, :2]),
            np.linalg.norm(dv, axis=1), _mahalanobis(dv, P[:, 2:, 2:]))


def _episode_metrics(world):
    """Episode sonu bilgi ölçütleri: (gerçek FIM logdet, belief logdet).

    gerçek FIM : tüm gözlemci izi + GERÇEK hedef yörüngesi üzerinden logdet(J).
                 Filtreden bağımsız — manevranın geometrik kalitesini ölçer, aşırı
                 güvenli/ıraksamış bir filtre tarafından şişirilemez.
    belief     : logdet(P⁻¹), yani RL ajanının ödülünün (BeliefLogDetReward) episode sonu
                 değeri — filtrenin KENDİ güveni. Gerçek FIM ile arasındaki fark
                 filtre tutarlılığının göstergesidir.
    """
    own = np.asarray(world.ownship.hist)
    t = world.dt * np.arange(len(own))
    fim = _fim_logdet(t, own, world.target.state, 0.0,
                      world.ownship.sensor.sigma, world.ownship.filter.propagate_state)
    P = world.ownship.filter.covariance
    sgn, ld = np.linalg.slogdet(np.linalg.inv(P + 1e-9 * np.eye(4)))   # BeliefLogDetReward ile aynı
    return fim, (ld if sgn > 0 else -50.0)


def monte_carlo(args):
    print(f"\nMonte Carlo — {args.episodes} episode (seed {args.seed0}..{args.seed0 + args.episodes - 1})")
    headings = {}          # (yöntem, karar anı) -> seçilen rotalar listesi
    rows, series = [], {}
    for name, key in METHODS:
        errs, fims, beliefs = [], [], []
        dEp, dMp, dEv, dMv = [], [], [], []
        for e in range(args.episodes):
            world, info = run_one(key, args, args.seed0 + e)
            errs.append(info["pos_err"])
            f, b = _episode_metrics(world)
            fims.append(f); beliefs.append(b)
            t_s, ep, mp, ev, mv = _error_series(world)
            dEp.append(ep); dMp.append(mp); dEv.append(ev); dMv.append(mv)
            for row in info["trace"]:
                headings.setdefault((name, round(row["t"])), []).append(row["action_deg"])
        rows.append((name, np.array(errs), np.mean(fims), np.mean(beliefs)))
        series[name] = (t_s, np.vstack(dEp), np.vstack(dMp),    # konum: hata, Mahalanobis
                        np.vstack(dEv), np.vstack(dMv))         # hız:   hata, Mahalanobis
        print(f"  · {name} bitti", flush=True)

    # logdet mutlak değerinin işareti birim artefaktıdır (metre/saniye seçimi belirler);
    # anlamlı olan yöntemler arası FARK. En kötü yöntemi 0 kabul edip farkları raporluyoruz.
    fim_ref = min(r[2] for r in rows)
    bel_ref = min(r[3] for r in rows)
    fim_worst = next(r[0] for r in rows if r[2] == fim_ref)

    print("\n  pos_err = sonuç (kestirim hatası) | Δ logdet = sebep (bilgi içeriği), en kötü = 0")
    print("  NOT: RH ve Kahin logdet J'yi (ufuk içinde) doğrudan optimize eder, RL ödülü de")
    print("  onun belief hâlidir — bu sütunlar tarafsız bir sıralama değil, teşhis aracıdır.")
    print(f"{'Yontem':12s} | {'ort':>9s} | {'medyan':>9s} | {'std':>9s} | {'maks':>9s} | "
          f"{'D logdet J':>10s} | {'eksen kat':>9s} | {'D logdet P^-1':>13s}")
    print("-" * 104)
    for name, errs, fim, bel in rows:
        d_fim = fim - fim_ref
        axis = np.exp(d_fim / 8.0)      # 4B elipsoidin eksen başına daralma katsayısı
        print(f"{name:12s} | {errs.mean():9.1f} | {np.median(errs):9.1f} | "
             f"{errs.std():9.1f} | {errs.max():9.1f} | {d_fim:10.2f} | {axis:8.2f}x | "
             f"{bel - bel_ref:13.2f}")
    print(f"  referans (en kötü): {fim_worst} → logdet J = {fim_ref:.2f}, "
          f"logdet P^-1 = {bel_ref:.2f}")
    print()
    if SHOW_DECISION_STATS:
        decision_stats(args, headings)
    return series, sorted({t for (_, t) in headings})


def draw_time_series(series, dec_t, n_ep):
    """MC zaman serileri — konum ve hız için doğruluk (RMSE) + tutarlılık (Ristic dM)."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 11.5), sharex=True, constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes
    for name, (t, dEp, dMp, dEv, dMv) in series.items():
        st = dict(color=COLORS[name], linestyle=LINESTYLES[name], lw=1.7, alpha=0.9, label=name)
        ax1.semilogy(t, np.sqrt((dEp ** 2).mean(axis=0)), **st)     # RMSE = sqrt(E[dE²])
        ax2.plot(t, dMp.mean(axis=0), **st)
        ax3.semilogy(t, np.sqrt((dEv ** 2).mean(axis=0)), **st)
        ax4.plot(t, dMv.mean(axis=0), **st)

    for ax in axes:
        for t in dec_t:
            ax.axvline(t, color="0.6", ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.grid(alpha=0.3, which="both")
    ax1.set_ylabel("Konum RMSE (m)")
    ax1.set_title(f"Konum doğruluğu — MC n={n_ep} (dikey çizgiler: karar anları)", fontsize=10)
    ax1.legend(fontsize=8, ncol=2)
    ax2.set_ylabel("Konum dM")
    ax2.set_title("Konum tutarlılığı — Ristic dM (ortalama Mahalanobis)", fontsize=10)
    ax3.set_ylabel("Hız RMSE (m/s)")
    ax3.set_title("Hız doğruluğu", fontsize=10)
    ax4.set_xlabel("t (s)"); ax4.set_ylabel("Hız dM")
    ax4.set_title("Hız tutarlılığı — ortalama Mahalanobis", fontsize=10)


def decision_stats(args, headings):
    """Karar anı bazında rota önerilerinin MC dağılımı + tek koşumla kıyası."""
    single = {}            # (yöntem, karar anı) -> görselleştirilen koşumun rotası
    for name, key in METHODS:
        _, info = run_one(key, args, args.plot_seed)
        for row in info["trace"]:
            single[(name, round(row["t"]))] = row["action_deg"]

    print(f"Rota önerisi değişkenliği — MC (n={args.episodes}) vs tek koşum "
          f"(seed={args.plot_seed}); açılar dairesel istatistik")
    print(f"{'Yontem':12s} | {'t':>5s} | {'MC ort':>7s} | {'MC std':>7s} | "
          f"{'tek kosum':>9s} | {'sapma':>6s}")
    print("-" * 62)
    for (name, t) in sorted(headings, key=lambda k: (METHOD_ORDER[k[0]], k[1])):
        mean, std = _circ_stats(headings[(name, t)])
        s = single.get((name, t))
        if s is None:
            print(f"{name:12s} | {t:5d} | {mean:7.1f} | {std:7.1f} | {'-':>9s} | {'-':>6s}")
            continue
        dev = abs(np.degrees(np.arctan2(np.sin(np.radians(s - mean)),
                                        np.cos(np.radians(s - mean)))))
        print(f"{name:12s} | {t:5d} | {mean:7.1f} | {std:7.1f} | {s % 360:9.1f} | {dev:6.1f}")
    print()


# ---------------- 2) Tek koşum ----------------
def _pack(world, info):
    own = np.asarray(world.ownship.hist)
    t = world.dt * np.arange(len(own))
    d = np.diff(own, axis=0)
    return dict(own=own, t=t, trace=info.get("trace", []),
               tgt=world.target.pos0 + np.outer(t, world.target.vel),
               hdg=np.degrees(np.arctan2(d[:, 0], d[:, 1])),
               est=world.ownship.filter.estimate, P=world.ownship.filter.covariance)


def cov_ellipse(ax, mean, P2, nsig=2.0, **kw):
    vals, vecs = np.linalg.eigh(P2)
    vals = np.maximum(vals, 1e-9)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nsig * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, **kw))


def print_decisions(runs):
    print(f"\nKarar anlari — seed={runs[0][1]['seed']}")
    print(f"{'Yontem':12s} | {'t':>6s} | {'kerteriz':>9s} | {'rota':>7s} | {'secilen':>8s}")
    print("-" * 55)
    for name, r in runs:
        for row in r["trace"]:
            print(f"{name:12s} | {row['t']:6.0f} | {row['bearing_deg']:9.1f} | "
                 f"{row['hdg_deg']:7.1f} | {row['action_deg']:8.1f}")
    print()


def _bearing_rays(ax, r, color, dt):
    """Karar anlarında gemiden hedefe kerteriz çizgileri. (meas_line, truth_line) döner.

    truth : gemi konumundan o andaki GERÇEK hedef konumuna düz çizgi (gri)
    meas  : sensörün ham ölçümü yönünde ışın (noktalı, yöntem renginde). Uzunluğu
            gerçek menzile eşitlenir — kerteriz-only ölçümde menzil bilgisi yoktur,
            bu yalnızca okunabilirlik içindir; ucun gerçek hedeften sapması = ölçüm hatası.
    """
    meas_line = truth_line = None
    for row in r["trace"]:
        own = row["own"]
        k = min(int(round(row["t"] / dt)), len(r["tgt"]) - 1)
        tgt = r["tgt"][k]
        if SHOW_TRUTH_RAYS:
            truth_line, = ax.plot([own[0], tgt[0]], [own[1], tgt[1]], '-',
                                  color="0.45", lw=0.8, alpha=0.7, zorder=1)
        if SHOW_MEAS_RAYS and row["z_deg"] is not None:
            th = np.radians(row["z_deg"])
            tip = own + np.linalg.norm(tgt - own) * np.array([np.sin(th), np.cos(th)])
            meas_line, = ax.plot([own[0], tip[0]], [own[1], tip[1]], ':', color=color,
                                 lw=1.2, alpha=0.75, zorder=2)
    return meas_line, truth_line


def _true_bearing_deg(r):
    """Her dt anında gemiden hedefe GERÇEK kerteriz (derece, kuzeyden saat yönü)."""
    d = r["tgt"] - r["own"]
    return np.degrees(np.arctan2(d[:, 0], d[:, 1]))


def _decision_times(runs):
    """Tüm yöntemlerin karar anlarının birleşimi (Kahin t=0'da da karar verir)."""
    return sorted({row["t"] for _, r in runs for row in r["trace"]})


def _vlines(ax, times):
    for t in times:
        ax.axvline(t, color="0.6", ls="--", lw=0.8, alpha=0.7, zorder=0)


def draw(runs, seed, dt):
    fig = plt.figure(figsize=(15.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1])
    axL = fig.add_subplot(gs[:, 0])                       # sol: XY izi (tam yükseklik)
    ax1 = fig.add_subplot(gs[0, 1])                       # sağ-1: rota
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)           # sağ-2: mesafe
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)           # sağ-3: gerçek kerteriz
    ax4 = fig.add_subplot(gs[3, 1], sharex=ax1)           # sağ-4: kerteriz değişimi
    dec_t = _decision_times(runs)

    tgt = runs[0][1]["tgt"]
    axL.plot(tgt[:, 0], tgt[:, 1], '-', color="k", lw=2.2, label="Gerçek hedef izi")
    axL.plot(*tgt[-1], 'k*', ms=14, label="Gerçek hedef (son)")
    meas = truth = None
    for name, r in runs:
        m, tr = _bearing_rays(axL, r, COLORS[name], dt)
        meas, truth = m or meas, tr or truth
    if truth is not None:
        truth.set_label("kerteriz — gerçek (karar anları)")
    if meas is not None:
        meas.set_label("kerteriz — ham ölçüm (karar anları)")
    for name, r in runs:
        c, ls, mk = COLORS[name], LINESTYLES[name], MARKERS[name]
        own = r["own"]
        axL.plot(own[:, 0], own[:, 1], color=c, lw=1.8, alpha=0.9, linestyle=ls,
                 marker=mk, markevery=max(1, len(own) // 6), ms=5, mfc="none",
                 label=f"{name} izi")
        axL.plot(*own[0], 'o', color=c, ms=5)
        axL.plot(*r["est"][:2], marker=mk, color=c, ms=9, mec="k", ls='none',
                 label=f"{name} kestirim")
        cov_ellipse(axL, r["est"][:2], r["P"][:2, :2], nsig=2.0, ec=c, lw=1.4, ls=ls)
    axL.set_xlabel("X (m)"); axL.set_ylabel("Y (m)")
    axL.set_title(f"XY iz + filtre kestirimi (2σ) — seed={seed}")
    axL.axis("equal"); axL.grid(alpha=0.3)
    axL.legend(fontsize=6.5, loc="best")

    # --- sağ-1: rota (heading) ---
    for name, r in runs:
        t_mid = 0.5 * (r["t"][:-1] + r["t"][1:])
        ax1.plot(t_mid, r["hdg"], color=COLORS[name], lw=1.6, alpha=0.9,
                 linestyle=LINESTYLES[name], marker=MARKERS[name], ms=4, mfc="none",
                 drawstyle="steps-mid", label=name)
    _vlines(ax1, dec_t)
    ax1.set_ylabel("Rota (°)")
    ax1.set_title("Rota (heading) — dikey çizgiler: karar anları", fontsize=10)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, loc="best", ncol=2)

    # --- sağ-2: gemi-hedef mesafesi ---
    for name, r in runs:
        rng = np.linalg.norm(r["tgt"] - r["own"], axis=1)
        ax2.plot(r["t"], rng, color=COLORS[name], lw=1.6, alpha=0.9,
                 linestyle=LINESTYLES[name], label=name)
    _vlines(ax2, dec_t)
    ax2.set_ylabel("Mesafe (m)")
    ax2.set_title("Gemi–hedef gerçek mesafesi", fontsize=10)
    ax2.grid(alpha=0.3)

    # --- sağ-3: gerçek kerteriz ---
    for name, r in runs:
        ax3.plot(r["t"], _true_bearing_deg(r), color=COLORS[name], lw=1.6, alpha=0.9,
                 linestyle=LINESTYLES[name], label=name)
    _vlines(ax3, dec_t)
    ax3.set_ylabel("Gerçek kerteriz (°)")
    ax3.set_title("Gemiden hedefe gerçek kerteriz", fontsize=10)
    ax3.grid(alpha=0.3)

    # --- sağ-4: kerteriz değişim hızı (gözlenebilirliğin doğrudan ölçüsü) ---
    for name, r in runs:
        brg_rad = np.unwrap(np.radians(_true_bearing_deg(r)))
        rate = np.degrees(np.gradient(brg_rad, r["t"]))                  # °/s
        ax4.plot(r["t"], rate, color=COLORS[name], lw=1.6, alpha=0.9,
                 linestyle=LINESTYLES[name], label=name)
    ax4.axhline(0.0, color="0.3", lw=0.8, alpha=0.6)
    _vlines(ax4, dec_t)
    ax4.set_xlabel("t (s)"); ax4.set_ylabel("Kerteriz değişimi (°/s)")
    ax4.set_title("Kerteriz değişim hızı — |büyük| = menzil daha hızlı çözülür", fontsize=10)
    ax4.grid(alpha=0.3)


def single_run(args):
    runs, dt = [], None
    for name, key in METHODS:
        if name in PLOT_SKIP:
            continue
        world, info = run_one(key, args, args.plot_seed)
        r = _pack(world, info)
        r["seed"] = args.plot_seed
        runs.append((name, r))
        dt = world.dt
    #print_decisions(runs)
    draw(runs, args.plot_seed, dt)


# ---------------- CLI ----------------
def main():
    # Dünya ayarları modelin sidecar'ından gelir; AYARLAR'daki OVERRIDES ve aşağıdaki
    # bayraklar onun ÜZERINE yazar (kasıtlı sapma). Bayraklar verilmezse kayma yapmaz.
    p = argparse.ArgumentParser(description="RL ajanları vs klasik planlayıcılar")
    p.add_argument("--model", default=None,
                   help="tek ajan kısayolu: AYARLAR'daki AGENTS listesinin yerine geçer")
    p.add_argument("--algo", choices=["ppo", "dqn", "npdqn"], default=None,
                   help="verilmezse modelin sidecar'ından okunur "
                        "(ppo/dqn = SB3 .zip, npdqn = tma/dqn.py .npz)")
    p.add_argument("--scenario", default=None, help="sapma: FixedScenario JSON yolu")
    p.add_argument("--obs", default=None, choices=["absolute", "los"], help="sapma")
    p.add_argument("--turn-rate-deg", type=float, default=None, help="sapma")
    p.add_argument("--no-opening-leg", action="store_true", help="sapma: açılış bacağını kapat")
    p.add_argument("--opening-offset-deg", type=float, default=None,
                   help="sapma: açılış bacağının kerterize ofseti (90 dik, 0 ilk temasa git)")
    p.add_argument("--total-time", type=float, default=None,
                   help="sapma: episode uzunluğu (s)")
    p.add_argument("--episodes", type=int, default=EPISODES, help="Monte Carlo episode sayısı (0 = atla)")
    p.add_argument("--seed0", type=int, default=SEED0, help="Monte Carlo ilk tohum")
    p.add_argument("--plot-seed", type=int, default=PLOT_SEED, help="görselleştirilecek koşum")
    p.add_argument("--no-plot", action="store_true", help="tek koşum görselleştirmesini atla")
    args = p.parse_args()

    # --- kıyaslanacak ajanlar: AYARLAR'daki AGENTS, --model verilirse onun yerine ---
    specs = ([(os.path.basename(args.model), args.model, args.algo)] if args.model
             else list(AGENTS))

    # --- kasıtlı sapmalar (AYARLAR + bayraklar) ---
    overrides = dict(OVERRIDES)
    for key, val in (("scenario", args.scenario), ("obs", args.obs),
                     ("turn_rate_deg", args.turn_rate_deg),
                     ("opening_offset_deg", args.opening_offset_deg),
                     ("total_time", args.total_time)):
        if val is not None:
            overrides[key] = val
    if args.no_opening_leg:
        overrides["opening_leg"] = False

    # --- her ajanın eğitim ayarını oku; ortak değerlendirme dünyasını kur ---
    trains = []
    for name, path, algo in specs:
        cfg, src = runconfig.load(path), runconfig.sidecar_path(path)
        if cfg is None:
            cfg = dict(FALLBACK_TRAIN)
            src = (f"{src} YOK -> FALLBACK_TRAIN varsayıldı "
                   f"(main.py ile yeniden eğitirseniz gerçek değerler kaydedilir)")
        trains.append((name, path, algo or cfg.get("algo") or ALGO, cfg, src))

    # Referans = ilk ajanın eğitim ayarı (ajan yoksa FALLBACK_TRAIN). Dünya TÜM
    # yöntemlerde ortaktır; ajanların yalnızca `obs`'u kendi eğitiminden gelir.
    ref_cfg = trains[0][3] if trains else dict(FALLBACK_TRAIN)
    world_cfg, _ = runconfig.resolve(ref_cfg, overrides)

    args.scenario = world_cfg["scenario"]
    args.turn_rate_deg = world_cfg["turn_rate_deg"]
    args.opening_offset_deg = world_cfg["opening_offset_deg"]
    args.total_time = world_cfg["total_time"]
    args.opening_leg = world_cfg["opening_leg"]
    args.obs = world_cfg["obs"]                 # yalnızca bilgi; ajanlar kendi obs'unu taşır
    args.plot = PLOT and not args.no_plot
    if args.plot_seed is None:
        args.plot_seed = args.seed0

    # --- ajanları yükle, kaymalarını raporla, METHODS'un başına ekle ---
    global METHOD_ORDER
    if any(algo != "npdqn" for _, _, algo, _, _ in trains):
        import torch
        torch.set_num_threads(1)  # tekrarlanabilirlik: çok-iş-parçacıklı toplama sırası
                                  # process'ten process'e değişiyor, sonuçlar kayıyordu

    for i, (name, path, algo, cfg, src) in enumerate(trains):
        if algo == "npdqn":                       # saf numpy ajan — torch/SB3 gerekmez
            from tma.dqn import DQN as NpDQN      # predict() cephesi SB3 ile aynı
            model = NpDQN.from_file(path + ".npz")
        else:
            from stable_baselines3 import PPO, DQN
            model = {"ppo": PPO, "dqn": DQN}[algo].load(path)

        # Dünya ortak; obs ajanın kendi eğitiminden. Kayma ajan BAŞINA raporlanır.
        agent_cfg = dict(world_cfg, obs=cfg.get("obs"))
        if overrides.get("obs") is not None:       # obs'u zorlamak ajanın girdisini bozar
            agent_cfg["obs"] = overrides["obs"]
        shifts = runconfig.resolve(cfg, agent_cfg)[1]
        runconfig.banner(f"{name}  ({path}, algo={algo})", cfg, agent_cfg, shifts, src)

        METHODS.insert(i, (name, AgentRun(model, agent_cfg["obs"])))
        COLORS[name] = _AG_COLORS[i % len(_AG_COLORS)]
        LINESTYLES[name] = _AG_LINESTYLES[i % len(_AG_LINESTYLES)]
        MARKERS[name] = _AG_MARKERS[i % len(_AG_MARKERS)]

    METHOD_ORDER = {name: i for i, (name, _) in enumerate(METHODS)}

    series = dec_t = None
    if args.episodes > 0:
        series, dec_t = monte_carlo(args)
    if args.plot:
        if series is not None and SHOW_TIME_SERIES:
            draw_time_series(series, dec_t, args.episodes)
        single_run(args)              # figürleri kurar
        plt.show()                    # tüm pencereler birlikte açılır


if __name__ == "__main__":
    main()
