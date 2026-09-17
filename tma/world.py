"""Platform + World — döngü ve gerçek durum (pluggable).

reward_fn / initializer / obs_encoder tak-çıkar. Bir apply(a) = bir KARAR aralığı =
plan_every ölçüm adımı; iç döngü _advance() (reset'teki açılış bacağı da onu kullanır).
opening_heading_fn verilirse açılış bacağı SÜREKLI bir başlıkla koşulur (makale: PTB);
verilmezse ilk kerterize opening_offset eklenip en yakın ayrık rotaya yuvarlanır
(Faz 2; 90° = kerterize dik, 0° = ilk temas kerterizine git — bkz. _opening_heading).
last_z son ölçümü saklar (Paper12Obs için).
"""
import numpy as np
from .util import wrap

DT = 10.0
T_TOTAL = 900.0
PLAN_EVERY = 30
OWN_SPD = 8.0
SIGMA_DEG = 1.0
N_HYP = 6
ACTIONS = np.deg2rad(np.arange(0, 360, 30))


class Platform:
    def __init__(self, pos, vel=None, hdg=None, speed=None, turn_rate_max=np.inf,
                 sensor=None, filter=None, maneuver=None, privileged=False):
        self.pos = np.asarray(pos, float)
        self.pos0 = self.pos.copy()
        self.vel = None if vel is None else np.asarray(vel, float)
        self.hdg = hdg
        self.speed = speed
        self.turn_rate_max = turn_rate_max
        self.sensor = sensor
        self.filter = filter
        self.maneuver = maneuver
        self.privileged = privileged
        self.hist = [self.pos.copy()]

    def pos_at(self, t):
        return self.pos0 + self.vel * t

    @property
    def state(self):
        return np.concatenate([self.pos0, self.vel])

    def propagate(self, dt, desired_hdg=None):
        if desired_hdg is not None:
            if np.isinf(self.turn_rate_max):
                self.hdg = desired_hdg
            else:
                dpsi = wrap(desired_hdg - self.hdg)
                # Dönüş YÖNÜ platformun kararı: kendi sensörünün son kerterizine bakıp
                # hedefe sırtını dönmeyecek tarafı seçer. En kısa yol hedefin tam tersi
                # açıdan (kerteriz+180) geçiyorsa, uzun yoldan dönülür.
                z = None if self.sensor is None else self.sensor.last_z
                if z is not None:
                    d_back = wrap(wrap(z + np.pi) - self.hdg)
                    if np.sign(d_back) == np.sign(dpsi) and abs(d_back) < abs(dpsi):
                        dpsi -= np.sign(dpsi) * 2 * np.pi
                step = self.turn_rate_max * dt
                self.hdg = wrap(self.hdg + np.clip(dpsi, -step, step))
        self.pos = self.pos + self.speed * np.array([np.sin(self.hdg), np.cos(self.hdg)]) * dt
        self.hist.append(self.pos.copy())


class World:
    def __init__(self, ownship, target, reward_fn, initializer, obs_encoder,
                 seed=0, dt=DT, total_time=T_TOTAL, plan_every=PLAN_EVERY,
                 actions=ACTIONS, opening_leg=False, opening_offset=np.pi / 2,
                 opening_heading_fn=None):
        self.ownship = ownship
        self.target = target
        self.reward_fn = reward_fn
        self.initializer = initializer
        self.obs_encoder = obs_encoder
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.total_time = total_time
        self.plan_every = plan_every
        self.actions = actions
        self.opening_leg = opening_leg
        self.opening_offset = opening_offset
        self.opening_heading_fn = opening_heading_fn
        self.t = 0.0
        self.k = 0
        self.last_z = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rg = self.rng
        spec = self.initializer.draw(rg)
        tgt = np.asarray(spec["target_state"], float)
        self.target.pos0 = tgt[:2].copy()
        self.target.pos = tgt[:2].copy()
        self.target.vel = tgt[2:].copy()
        self.target.hist = [self.target.pos.copy()]
        self.ownship.pos = np.asarray(spec["own_pos"], float).copy()
        self.ownship.pos0 = self.ownship.pos.copy()
        self.ownship.hist = [self.ownship.pos.copy()]
        self.ownship.hdg = float(spec["own_hdg"])
        if spec.get("own_speed") is not None:      # senaryo hız veriyorsa onu kullan
            self.ownship.speed = float(spec["own_speed"])
        self.t = 0.0
        self.k = 0
        z0 = self.ownship.sensor.sample(self.ownship.pos, self.target.state, 0.0, rg)
        self.last_z = z0
        self.ownship.filter.init(z0, self.ownship.pos)
        # Kestirim geçmişi: her dt'de (mean, cov). Hesabı etkilemez, yalnızca zaman-serisi
        # teşhisi (RMSE/dE/dM) için kayıt — bkz. analyze.py.
        self.est_hist = [self.ownship.filter.estimate.copy()]
        self.P_hist = [self.ownship.filter.covariance.copy()]
        if self.opening_leg and not self._privileged():
            self._advance(self._opening_heading())
        self._prev_reward = self.reward_fn.baseline(self)
        return self.observe()

    def _privileged(self):
        m = self.ownship.maneuver
        return m is not None and getattr(m, "privileged", False)

    def _opening_heading(self):
        """Açılış bacağı rotası = ilk kerteriz + opening_offset, en yakın ayrık aksiyona
        yuvarlanmış. bearing0 açılış anındaki BELIEF kerterizidir; t=0'da hipotezlerin
        tamamı z0 ışını üzerinde olduğu için ham ölçüme eşittir.

        opening_offset = +pi/2 (varsayılan): kerterize DİK. LOS'a maksimum enine bileşen →
            kendi-kaynaklı bearing rate en yüksek → menzili en hızlı çözen açılış (Karar C).
        opening_offset = 0: İLK TEMAS KERTERİZİNE GİT. Hareket LOS boyunca; kendi
            hareketinin ürettiği enine bileşen ~0, dolayısıyla bearing rate yalnızca
            hedefin kendi hareketinden gelir. Gözlenebilirlik açısından kasıtlı olarak
            zayıf bir kıyas tabanı — dik açılışın kazancını ölçmek için.
        """
        if self.opening_heading_fn is not None:
            return self.opening_heading_fn(self)          # sürekli (makale: PTB)
        m = self.ownship.filter.estimate
        rel = m[:2] - self.ownship.pos
        bearing0 = np.arctan2(rel[0], rel[1])
        desired = wrap(bearing0 + self.opening_offset)
        a = int(np.argmin(np.abs(wrap(self.actions - desired))))
        return self.actions[a]

    def _advance(self, desired_hdg):
        for _ in range(self.plan_every):
            self.k += 1
            self.t = self.k * self.dt
            self.ownship.propagate(self.dt, desired_hdg)
            z = self.ownship.sensor.sample(self.ownship.pos, self.target.state, self.t, self.rng)
            self.last_z = z
            self.ownship.filter.step(z, self.ownship.pos)
            self.est_hist.append(self.ownship.filter.estimate.copy())
            self.P_hist.append(self.ownship.filter.covariance.copy())
            if self.t >= self.total_time:
                break

    def apply(self, action):
        # action: manevra continuous ise doğrudan RADYAN açı, değilse actions[] indeksi
        # (bkz. Maneuver.continuous). RL yolunda maneuver=None -> her zaman indeks.
        m = self.ownship.maneuver
        self._advance(action if (m is not None and m.continuous) else self.actions[action])
        reward = self.reward_fn(self, self._prev_reward, action)
        self._prev_reward = self.reward_fn.baseline(self)
        done = self.t >= self.total_time
        tp = self.target.pos_at(self.t)
        err = np.linalg.norm(self.ownship.filter.estimate[:2] - tp)
        return self.observe(), reward, done, {"pos_err": err}

    def true_target_state(self):
        return np.concatenate([self.target.pos_at(self.t), self.target.vel])

    def maneuver_context(self):
        from .maneuver import ManeuverContext
        o = self.ownship
        priv = self._privileged()
        f = o.filter
        return ManeuverContext(
            own=o.pos, hdg=o.hdg, own_hist=o.hist, t=self.t, dt=self.dt,
            own_speed=o.speed, sigma=o.sensor.sigma, action_set=self.actions,
            estimate=f.estimate, covariance=f.covariance,
            propagate_state=f.propagate_state, planning_states=f.planning_states,
            plan_horizon=self.plan_every * self.dt, rng=self.rng,
            true_target_state=(self.true_target_state if priv else None))

    def observe(self):
        return self.obs_encoder(self)
