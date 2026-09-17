"""Manevra katmanı — decide(ctx) nesneleri.

FIM planlayıcılar ctx.planning_states() (ağırlıklı hipotezler) + ctx.propagate_state
kullanır. Kahin ctx.true_target_state ayrıcalıklı kanalını kullanır. PTB/ITO
baseline'ları ctx.estimate / ctx.covariance (belief) kullanır.
"""
from dataclasses import dataclass
import numpy as np
from .util import wrap


@dataclass
class ManeuverContext:
    own: np.ndarray
    hdg: float
    own_hist: list
    t: float
    dt: float
    own_speed: float
    sigma: float
    action_set: np.ndarray
    estimate: np.ndarray          # filtre kestirimi (mean) — PTB/ITO kullanır
    covariance: np.ndarray        # filtre kovaryansı — ITO prior'u kullanır
    propagate_state: object       # ileri-model (filtreden)
    planning_states: object       # [(state, weight)] ağırlıklı hipotezler
    plan_horizon: float           # bir karar aralığının süresi (s)
    rng: object
    true_target_state: object     # None (dürüst); yalnızca privileged manevrada dolu


# ================= FIM yardımcıları (RH/RH2) =================
def _fim_logdet(all_t, all_xy, tgt_state, t_now, sigma, propagate_state):
    tp = propagate_state(tgt_state, all_t - t_now)[:, :2]
    d = tp - all_xy
    r2 = np.sum(d**2, axis=1)
    c, s = d[:, 1] / r2, -d[:, 0] / r2
    Hm = np.column_stack([c, s, all_t * c, all_t * s])
    sgn, ld = np.linalg.slogdet(Hm.T @ Hm / sigma**2)
    return ld if sgn > 0 else -1e3


def _own_future(ctx, turn_delay, hdg2, horizon):
    n = int(horizon / ctx.dt)
    v1 = ctx.own_speed * np.array([np.sin(ctx.hdg), np.cos(ctx.hdg)])
    v2 = ctx.own_speed * np.array([np.sin(hdg2), np.cos(hdg2)])
    ts = ctx.dt * np.arange(1, n + 1)
    xy = np.where(ts[:, None] <= turn_delay,
                  ctx.own + np.outer(ts, np.ones(2)) * v1,
                  ctx.own + turn_delay * v1 + np.outer(ts - turn_delay, np.ones(2)) * v2)
    return ctx.t + ts, xy


def _score(ctx, fut_t, fut_xy, tgt_states, weights):
    past_t = ctx.dt * np.arange(len(ctx.own_hist))
    past_xy = np.asarray(ctx.own_hist)
    all_t = np.concatenate([past_t, fut_t])
    all_xy = np.vstack([past_xy, fut_xy])
    return sum(w * _fim_logdet(all_t, all_xy, x, ctx.t, ctx.sigma, ctx.propagate_state)
               for w, x in zip(weights, tgt_states))


# ================= perpendicular-to-bearing yardımcısı =================
TIE_SIDE = +1          # beraberlikte kerterizin hangi tarafı: +1 = bearing+90


def ptb_heading(mean, own, own_speed, dt, horizon):
    """Makalenin AÇILIŞ bacağı (leg 1) rotası: kerterize dik iki yönden birini seç.

    Kural (makale): "the one that keeps the observer oriented toward the predicted
    target position (accounting for estimated target velocity)".

    DIKKAT — bu kural açılış anında MATEMATIKSEL OLARAK ATILDIR: CKF hızı tam sıfırla
    başlatır (filter_ckf.CubatureKF.init, makale eq 14 "initial relative velocity is
    set to zero"), dolayısıyla tahmini hedef konumu = kestirim konumu ve iki aday da
    ona TAM DIK -> her iki skor da analitik olarak 0. Makale bu beraberlik için bir
    kural vermiyor (gerçek bir boşluk).

    Eskiden skorlar ~1e-15 kayan-nokta gürültüsüyle karşılaştırılıyordu, yani açılış
    bacağının sağ/sol tercihi yuvarlama hatasıyla belirleniyordu. Artık beraberlik
    TIE_SIDE ile deterministik çözülüyor; kural anlamlı hâle geldiğinde (hız kestirimi
    sıfırdan farklıysa) yine skorlar karar veriyor.

    Yalnızca paper_repro tarafından kullanılır (opening_heading_fn); tez dünyası
    açılış bacağını registry/build_world'ün ayrık ofset kuralıyla koşar.
    """
    rel = mean[:2] - own
    bearing = np.arctan2(rel[0], rel[1])
    tgt_future = mean[:2] + mean[2:] * horizon           # tahmini hedef konumu
    h1, h2 = wrap(bearing + np.pi / 2), wrap(bearing - np.pi / 2)
    s1, s2 = (float(np.dot(np.array([np.sin(h), np.cos(h)]), tgt_future - own))
              for h in (h1, h2))
    scale = max(float(np.linalg.norm(tgt_future - own)), 1e-12)
    if abs(s1 - s2) < 1e-9 * scale:                      # beraberlik -> sabit taraf
        return h1 if TIE_SIDE > 0 else h2
    return h1 if s1 > s2 else h2


# ================= manevralar =================
class Maneuver:
    privileged = False
    # continuous: decide() ne döndürüyor?
    #   False (varsayılan) -> ctx.action_set'e INDEKS (int). World.apply() actions[i] ile
    #                         açıya çevirir. RL de bu yolu kullanır (TMAGym Discrete(n)).
    #   True               -> doğrudan RADYAN açı (float); ayrık ızgaraya yuvarlama yapılmaz.
    # NOT: True yapmak kıyas adaletini etkiler — RL 12 ayrık rotayla sınırlıyken sürekli
    # açı kullanan manevra daha ince nişan alabilir (bkz. MIMARI_KARARLAR.md, ManeuverContext
    # "RH ve RL AYNI seti görür"). Karşılaştırma koşarken bayrağı bilinçli seçin.
    continuous = False

    def decide(self, ctx):
        raise NotImplementedError


class OracleManeuver(Maneuver):
    privileged = True

    def __init__(self, horizon=300.0):
        self.horizon = horizon

    def decide(self, ctx):
        tgt_now = ctx.true_target_state()
        best, best_a = -np.inf, 0
        for a, h in enumerate(ctx.action_set):
            ft, fxy = _own_future(ctx, 0.0, h, self.horizon)
            sc = _score(ctx, ft, fxy, [tgt_now], [1.0])
            if sc > best:
                best, best_a = sc, a
        return best_a


def _fim_targets(ctx, use_hypotheses):
    """FIM'in üzerinden alınacağı hedef durumları + ağırlıkları.

    use_hypotheses=False (varsayılan): filtrenin TEK, ağırlıklı (moment-eşlenmiş)
        kestirimi — certainty-equivalent planlama. RP-EKF bankasının 6 hipotezi
        planlamaya girmez, yalnızca rapor edilen ortalama kullanılır.
    use_hypotheses=True: ctx.planning_states() ile ağırlıklı hipotezler üzerinden
        beklenen logdet (MIMARI_KARARLAR.md "Karar A" seçenek (i) — eski davranış).
        Kıyas koşmak için True yapın.
    """
    if not use_hypotheses:
        return [ctx.estimate], [1.0]
    states = ctx.planning_states()
    return [s for s, _ in states], [w for _, w in states]


class RHManeuver(Maneuver):
    def __init__(self, horizon=300.0, use_hypotheses=False):
        self.horizon = horizon
        self.use_hypotheses = use_hypotheses

    def decide(self, ctx):
        tgt_states, weights = _fim_targets(ctx, self.use_hypotheses)
        best, best_a = -np.inf, 0
        for a, h in enumerate(ctx.action_set):
            ft, fxy = _own_future(ctx, 0.0, h, self.horizon)
            sc = _score(ctx, ft, fxy, tgt_states, weights)
            if sc > best:
                best, best_a = sc, a
        return best_a


class RH2Maneuver(Maneuver):
    def __init__(self, horizon=420.0, turn_delays=(0., 60., 120., 180.), use_hypotheses=False):
        self.horizon = horizon
        self.turn_delays = np.array(turn_delays)
        self.use_hypotheses = use_hypotheses

    def decide(self, ctx):
        tgt_states, weights = _fim_targets(ctx, self.use_hypotheses)
        best, best_first = -np.inf, 0
        cur = int(np.argmin(np.abs(wrap(ctx.action_set - ctx.hdg))))
        for td in self.turn_delays:
            for a, h in enumerate(ctx.action_set):
                ft, fxy = _own_future(ctx, td, h, self.horizon)
                sc = _score(ctx, ft, fxy, tgt_states, weights)
                if sc > best:
                    best, best_first = sc, (a if td == 0 else cur)
        return best_first


class RandomManeuver(Maneuver):
    def decide(self, ctx):
        return int(ctx.rng.integers(len(ctx.action_set)))


class StraightManeuver(Maneuver):
    def decide(self, ctx):
        return int(np.argmin(np.abs(wrap(ctx.action_set - ctx.hdg))))


# ================= makale baseline'ları =================
class PTBManeuver(Maneuver):
    """Perpendicular-to-bearing (genel dünya kuralı — makale açılışındaki ptb_heading'den
    FARKLI, bkz. not): kerterize göre ±offset_deg konumundaki iki adaydan, mevcut rotaya
    (ctx.hdg) açısal olarak UZAK olanı seçer. Tek kural bu; hangi yönden (sağdan/soldan)
    dönüleceği platformun kendi işi (bkz. Platform.propagate).

    offset_deg: kerterize göre ofset (varsayılan 90 = klasik dik-kerteriz). 90'dan farklı
    değerler denemek için varyant kurun: PTBManeuver(offset_deg=80). Adaylar bearing±offset
    olduğundan offset≠90'da ikisi 180° aralıklı değildir — "uzak olanı seç" kuralı yine geçerli.

    Not: makalenin açılış bacağında kullanılan ptb_heading() kasıtlı olarak değiştirilmedi
    (makale reprodüksiyonunu bozmamak için) — bu kural yalnızca bu sınıfa özel."""

    # PTB doğası gereği sürekli bir açı üretir (kerteriz ± 90°); ayrık ızgaraya yuvarlamak
    # onu yapay olarak sakatlıyordu (30°'lik kovada kalıp üst üste aynı aksiyonu seçiyordu).
    # False yapılırsa eski davranışa (en yakın ayrık aksiyon) döner — kıyas için bkz.
    # Maneuver.continuous notu.
    continuous = True

    def __init__(self, offset_deg=90.0):
        self.offset = np.deg2rad(offset_deg)
        self.offset_deg = offset_deg

    def decide(self, ctx):
        rel = ctx.estimate[:2] - ctx.own
        bearing = np.arctan2(rel[0], rel[1])
        h1 = wrap(bearing + self.offset)
        h2 = wrap(bearing - self.offset)
        h = h1 if abs(wrap(h1 - ctx.hdg)) > abs(wrap(h2 - ctx.hdg)) else h2
        if not self.continuous:
            return int(np.argmin(np.abs(wrap(ctx.action_set - h))))
        return h


class PaperPTBManeuver(Maneuver):
    """Ristic & Arulampalam'ın PTB baseline'ı — makale metnindeki kural birebir.

    "selects the heading most nearly perpendicular to the line of sight toward the
     CKF position estimate. Of the two candidate perpendicular directions, the one
     that keeps the observer oriented toward the predicted target position
     (accounting for estimated target velocity) is chosen."

    KRITIK: makalenin durumu GÖRELİdir (eq 2: x = x_hedef - x_gözlemci), dolayısıyla
    "predicted target position" = p_rel + (v_hedef - v_gözlemci)·MT. Gözlemci hız
    terimi ihmal edilemez: leg 1 zaten LOS'a dik koşulduğu için v_gözlemci iki adaya
    paralel/anti-paraleldir, yani seçimde BASKIN terimdir. Mutlak hızla hesaplayan
    sürüm episode'ların ~%43'ünde farklı aksiyon seçer ve ortalama dE'yi 5.6'ya
    çıkarır (makale 3.34); göreli hızla 2.6'ya iner ve std/maks/dM makaleyle örtüşür.

    `PTBManeuver`'dan AYRI tutuldu: o sınıf tez tarafının (analyze.py PTB varyantları)
    kuralıdır ve MIMARI_KARARLAR §11.B′ ile bilinçli olarak farklı tanımlanmıştır.

    continuous=False: makale aksiyon kümesinden seçiyor ("selects the heading",
    A = 16 rota) ve DQN/ITO de aynı ızgarada — kıyas adaleti için ayrık. Sürekli
    açıyla fark ölçülebilir değil (eşli fark ~0.005 dE).
    """
    continuous = False

    def decide(self, ctx):
        rel = ctx.estimate[:2] - ctx.own
        bearing = np.arctan2(rel[0], rel[1])
        own_vel = ctx.own_speed * np.array([np.sin(ctx.hdg), np.cos(ctx.hdg)])
        pred = rel + (ctx.estimate[2:] - own_vel) * ctx.plan_horizon   # göreli çerçeve
        h1, h2 = wrap(bearing + np.pi / 2), wrap(bearing - np.pi / 2)
        s1, s2 = (float(np.dot(np.array([np.sin(h), np.cos(h)]), pred))
                  for h in (h1, h2))
        scale = max(float(np.linalg.norm(pred)), 1e-12)
        if abs(s1 - s2) < 1e-9 * scale:
            h = h1 if TIE_SIDE > 0 else h2
        else:
            h = h1 if s1 > s2 else h2
        if self.continuous:
            return h
        return int(np.argmin(np.abs(wrap(ctx.action_set - h))))


class ITOManeuver(Maneuver):
    """Information-Theoretic Observer (D-optimal, konum FIM'i + prior).
    J = [P^p]^-1 + sum_j H_j^T H_j / sigma^2,  H_j = [-dy/r^2, dx/r^2] (konum gradyanı).
    Makale eq (22): mevcut konum bilgi matrisini prior alır, gelecek FIM'i ekler."""

    def decide(self, ctx):
        n = int(ctx.plan_horizon / ctx.dt)
        Ppos = ctx.covariance[:2, :2]
        J_prior = np.linalg.inv(Ppos + 1e-12 * np.eye(2))
        ts = ctx.dt * np.arange(1, n + 1)
        best, best_a = -np.inf, 0
        for a, h in enumerate(ctx.action_set):
            v = ctx.own_speed * np.array([np.sin(h), np.cos(h)])
            own_fut = ctx.own + np.outer(ts, v)                       # leg2 kendi izi
            tgt_fut = ctx.estimate[:2] + np.outer(ts, ctx.estimate[2:])  # tahmini hedef
            d = tgt_fut - own_fut
            r2 = np.sum(d**2, axis=1)
            Hx = -d[:, 1] / r2                                        # -dy/r^2
            Hy = d[:, 0] / r2                                         #  dx/r^2
            J = J_prior + np.array([[np.sum(Hx * Hx), np.sum(Hx * Hy)],
                                    [np.sum(Hx * Hy), np.sum(Hy * Hy)]]) / ctx.sigma**2
            sgn, ld = np.linalg.slogdet(J)
            sc = ld if sgn > 0 else -np.inf
            if sc > best:
                best, best_a = sc, a
        return best_a
