"""A: Nonlinear fonksiyon regresyonu — dqn.py'deki MLP + Adam ile.

Amaç: DQN'in sinir ağı çekirdeğini (MLP.forward / MLP.backward / Adam) DQN
mantığından SOYUTLAYIP saf bir regresyonda görmek.

Paralellik (dqn.py train_step ile):
    DQN:        err = Q[idx,A] - R       ; hedef = bilinmeyen ödül r
    Buradaki:   err = pred      - y      ; hedef = bilinen f(x) = sin(x)
Yani tek-karar DQN zaten bir regresyondu; burada "doğru cevabı" bildiğimiz için
ağın öğrenip öğrenmediğini gözle görebiliyoruz. MLP ve Adam sınıflarına HİÇ
dokunmuyoruz.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")               # başsız ortamda dosyaya çizim
import matplotlib.pyplot as plt

from tma.dqn import MLP, Adam           # <-- dqn.py'den birebir çekirdek


# ----------------------------------------------------------------------
# 1) Öğreneceğimiz nonlinear fonksiyon
# ----------------------------------------------------------------------
def target_fn(x):
    return np.sin(x)                # istersen sin(3x)*exp(-x/3) gibi değiştir

X_LO, X_HI = -2 * np.pi, 2 * np.pi
rng = np.random.default_rng(0)


def make_data(n):
    x = rng.uniform(X_LO, X_HI, size=(n, 1))     # şekil (n, 1) — girdi tek boyut
    y = target_fn(x)                             # şekil (n, 1)
    return x, y


# ----------------------------------------------------------------------
# 2) Veri + girdi normalizasyonu
#    He init makul girdi ölçeği varsayar; ham x ~[-6,6] yerine standartlaştır.
# ----------------------------------------------------------------------
x_train, y_train = make_data(1024)
x_mu, x_sd = x_train.mean(), x_train.std()
xn_train = (x_train - x_mu) / x_sd


def normalize(x):
    return (x - x_mu) / x_sd


# ----------------------------------------------------------------------
# 3) Ağ + optimizer  (1 -> 64 -> 64 -> 1, gizli ReLU, son katman lineer)
# ----------------------------------------------------------------------
net = MLP([1, 64, 64, 1], rng)
opt = Adam(net.P, lr=1e-3, clip=1.0)   # dqn ile aynı Adam; sadece lr'yi yükselttik

BATCH = 64
STEPS = 5000
loss_hist = []

for step in range(STEPS):
    # --- minibatch örnekle (replay buffer.sample'ın regresyon karşılığı) ---
    idx = rng.integers(0, len(xn_train), size=BATCH)
    xb, yb = xn_train[idx], y_train[idx]

    # --- ileri geçiş ---
    pred, cache = net.forward(xb)           # (BATCH, 1)

    # --- MSE kaybı ve gradyanı ---
    err = pred - yb                          # DQN'deki (qa - R) ile aynı rol
    loss = np.mean(err ** 2)
    dout = 2.0 * err / BATCH                  # dL/dpred ; /BATCH = mean için

    # --- geri geçiş + parametre güncelleme (dqn.py'nin kendi kodu) ---
    grads = net.backward(cache, dout)
    opt.step(net.P, grads)

    loss_hist.append(loss)
    if step % 500 == 0:
        print(f"step {step:5d}   MSE = {loss:.5f}")


# ----------------------------------------------------------------------
# 4) Değerlendirme (taze test kümesi) + görselleştirme
# ----------------------------------------------------------------------
x_test, y_test = make_data(2000)
pred_test, _ = net.forward(normalize(x_test))
test_mse = np.mean((pred_test - y_test) ** 2)
print(f"\nTest MSE = {test_mse:.5f}")

# yoğun ızgarada gerçek eğri vs ağ tahmini
xg = np.linspace(X_LO, X_HI, 400).reshape(-1, 1)
yg_true = target_fn(xg)
yg_pred, _ = net.forward(normalize(xg))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.scatter(x_train, y_train, s=6, alpha=0.15, label="eğitim verisi")
ax1.plot(xg, yg_true, "k--", lw=2, label="gerçek  f(x)=sin(x)")
ax1.plot(xg, yg_pred, "r-", lw=2, label="ağ tahmini")
ax1.set_title(f"Fit  (test MSE = {test_mse:.4f})")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.legend()

ax2.plot(loss_hist, lw=1)
ax2.set_yscale("log")
ax2.set_title("Eğitim kaybı (MSE, log)")
ax2.set_xlabel("adım"); ax2.set_ylabel("MSE")

plt.tight_layout()
plt.savefig("fit.png", dpi=110)
print("Grafik kaydedildi: fit.png")
