"""Sıfırdan (numpy) DQN — SB3 yok.

Mimari: obs -> 256 -> 128 -> 64 -> n_act (ReLU), Adam, Huber Q-kaybı, global-norm
gradyan kırpma (1.0), epsilon-greedy.

İKI KURGUYA BIRDEN hizmet eder; fark yalnızca geçişin `done` bayrağında:

  makale (paper_repro) : episode başına TEK karar -> her geçiş terminal (done=True)
      -> hedef = r. Bootstrap yok, gamma etkisiz, hedef ağı atıl. Makale de böyle
      ("with only one decision per episode, no discount factor is required").

  tez (main.py/analyze) : episode başına 4-5 karar, her kararda ödül (BeliefLogDetReward)
      -> hedef = r + gamma * (1-done) * max_a' Q_hedef(s', a'). Hedef ağı ARTIK gerekli.

Terminal geçişte (1-done)=0 olduğundan genel formül kendiliğinden hedef=r'ye iner:
tek sınıf iki dünyayı da doğru öğrenir. Bu yüzden `buf.add(s, a, r)` (s2/done
verilmezse terminal varsayılır) makale yolunda aynen çalışmaya devam eder.

SB3 uyumlu `predict()` cephesi var -> tma/rollout.py bu ajanı SB3 modeliyle aynı
şekilde koşturur. `save/load` ağ mimarisini de yazar; `DQN.from_file(path)` yükler.
"""
import numpy as np


def _he(shape, rng):
    return rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])


class MLP:
    def __init__(self, sizes, rng=None):
        """rng=None -> sıfır ağırlık (hedef ağı için; RNG akışını TÜKETMEZ).

        Hedef ağını rng ile kurmak eğitim tohumunu kaydırır ve mevcut makale
        sonuçlarını bozardı — bu yüzden sıfırla kurulup net'ten kopyalanır.
        """
        self.P = {}
        for i in range(len(sizes) - 1):
            self.P[f"W{i}"] = (np.zeros((sizes[i], sizes[i + 1])) if rng is None
                               else _he((sizes[i], sizes[i + 1]), rng))
            self.P[f"b{i}"] = np.zeros(sizes[i + 1])
        self.n_layers = len(sizes) - 1

    def forward(self, x):
        cache = {"a0": x}
        a = x
        for i in range(self.n_layers):
            z = a @ self.P[f"W{i}"] + self.P[f"b{i}"]
            cache[f"z{i}"] = z
            a = np.maximum(z, 0.0) if i < self.n_layers - 1 else z   # son katman lineer
            cache[f"a{i+1}"] = a
        return a, cache

    def backward(self, cache, dout):
        grads = {}
        da = dout
        for i in reversed(range(self.n_layers)):
            if i < self.n_layers - 1:
                da = da * (cache[f"z{i}"] > 0)
            grads[f"W{i}"] = cache[f"a{i}"].T @ da
            grads[f"b{i}"] = da.sum(0)
            da = da @ self.P[f"W{i}"].T
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        gn = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        scale = min(1.0, self.clip / (gn + 1e-12))
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class ReplayBuffer:
    """(s, a, r, s', done). s2/done verilmezse geçiş TERMINAL sayılır (makale kurgusu)."""

    def __init__(self, cap, obs_dim, rng):
        self.cap, self.rng = cap, rng
        self.s = np.zeros((cap, obs_dim))
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap)
        self.s2 = np.zeros((cap, obs_dim))
        self.d = np.zeros(cap)
        self.n = 0
        self.ptr = 0

    def add(self, s, a, r, s2=None, done=True):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r
        self.s2[i] = s if s2 is None else s2      # terminalde okunmaz, (1-done)=0
        self.d[i] = float(done)
        self.ptr = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQN:
    def __init__(self, obs_dim=12, n_act=16, hidden=(256, 128, 64), lr=3e-4,
                 buffer_cap=200_000, batch=256, huber_delta=1.0, gamma=0.99,
                 tau=0.01, eps_predict=0.05, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sizes = [obs_dim] + list(hidden) + [n_act]
        self.net = MLP(self.sizes, self.rng)
        self.tgt = MLP(self.sizes)                 # rng tüketmez
        self._sync_target(1.0)
        self.opt = Adam(self.net.P, lr=lr, clip=1.0)
        self.buf = ReplayBuffer(buffer_cap, obs_dim, self.rng)
        self.batch = batch
        self.delta = huber_delta
        self.gamma = gamma
        self.tau = tau
        self.eps = eps_predict
        self.n_act = n_act

    def _sync_target(self, tau):
        """Polyak: tgt <- tau*net + (1-tau)*tgt. tau=1.0 sert kopya."""
        for k, v in self.net.P.items():
            self.tgt.P[k] = v.copy() if tau >= 1.0 else tau * v + (1.0 - tau) * self.tgt.P[k]

    # ---- eylem seçimi ----
    def act(self, obs, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_act))
        q, _ = self.net.forward(obs[None])
        return int(np.argmax(q[0]))

    def greedy(self, obs):
        q, _ = self.net.forward(obs[None])
        return int(np.argmax(q[0]))

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        """SB3 uyumlu cephe — tma/rollout.run_agent_episode bunu çağırır."""
        o = np.asarray(obs, dtype=float).reshape(-1)
        return (self.greedy(o) if deterministic else self.act(o, self.eps)), None

    # ---- öğrenme ----
    def train_step(self):
        if self.buf.n < self.batch:
            return 0.0
        S, A, R, S2, D = self.buf.sample(self.batch)
        bootstrap = bool(np.any(D < 1.0))          # hepsi terminalse hedef zaten r
        if bootstrap:
            q2, _ = self.tgt.forward(S2)
            y = R + self.gamma * (1.0 - D) * q2.max(axis=1)
        else:
            y = R                                  # tek-karar terminal (makale)
        Q, cache = self.net.forward(S)
        idx = np.arange(len(A))
        err = Q[idx, A] - y
        # Huber gradyanı: |err|<=delta ise err, aksi halde delta*sign
        g = np.clip(err, -self.delta, self.delta)
        dQ = np.zeros_like(Q)
        dQ[idx, A] = g / len(A)
        grads = self.net.backward(cache, dQ)
        self.opt.step(self.net.P, grads)
        if bootstrap:
            self._sync_target(self.tau)            # atılken senkron da gereksiz
        huber = np.where(np.abs(err) <= self.delta, 0.5 * err**2,
                         self.delta * (np.abs(err) - 0.5 * self.delta))
        return float(huber.mean())

    # ---- kalıcılık ----
    def save(self, path):
        """Ağırlıklar + mimari. _sizes sayesinde yükleyen taraf obs_dim/n_act tahmin etmez."""
        np.savez(path, _sizes=np.asarray(self.sizes, dtype=np.int64),
                 _gamma=np.asarray([self.gamma], dtype=float),
                 **{k: v for k, v in self.net.P.items()})

    def load(self, path):
        d = np.load(path)
        for k in self.net.P:
            self.net.P[k] = d[k]
        self._sync_target(1.0)

    @classmethod
    def from_file(cls, path, **kw):
        """Mimariyi dosyadan okuyup ajanı kurar ve yükler.

        _sizes yoksa (bu sürümden önce kaydedilmiş ajanlar) ağırlık şekillerinden
        çıkarılır — models/agent_b0*.npz yeniden eğitilmeden yüklenebilir.
        """
        d = np.load(path)
        if "_sizes" in d.files:
            sizes = [int(v) for v in d["_sizes"]]
        else:
            n = sum(1 for k in d.files if k.startswith("W"))
            sizes = [int(d["W0"].shape[0])] + [int(d[f"W{i}"].shape[1]) for i in range(n)]
        agent = cls(obs_dim=sizes[0], n_act=sizes[-1], hidden=tuple(sizes[1:-1]), **kw)
        agent.load(path)
        return agent
