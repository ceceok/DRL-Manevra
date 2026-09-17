# TMA Yeniden Yapılandırma — Mimari Karar Dosyası

Bu dosya, bearings-only TMA kod tabanının pluggable (tak-çıkar) bir mimariye
taşınması için alınan tasarım kararlarını kaydeder. Amaç: temiz bir sayfadan
ilk versiyonu (v1) yazmak, mevcut davranışı bozmadan. VS Code'a taşınacak
referans belgedir.

---

## 0. Amaç ve kapsam

- **Problem:** tek gözlemci, yalnızca kerteriz (bearing) ölçümüyle hareketli
  hedef kestirimi; gözlenebilirlik kendi-manevraya bağlı.
- **Hedef:** sensör / filtre / manevra / ödül / senaryo-başlatıcıyı birbirinden
  bağımsız, tak-çıkar bileşenler yapmak; RL ile klasik planlayıcıları (RH, kahin)
  *aynı* dünyada, adil şekilde kıyaslamak.
- **v1 sınırı:** bearings-only, her yönden temas (baffle yok), tek hedef, CV
  hareket modeli, ani dönüş. Bunların hepsi arayüzde genişletilebilir bırakıldı.

---

## 1. Genel yapı (özet)

```
Sürücüler:   SB3 eğitim (Durum 1)      Kıyas döngüsü (Durum 2)
                   │  (TMAGym adaptörü)        │  (doğrudan)
                   └──────── reset() / apply() ┘
                                  ▼
World  ── gerçek durum · saat · döngü · reward + initializer (pluggable)
  ├── Hedef (Platform)   — kinematik, pasif
  └── Ownship (Platform) — durum · iz · kinematik
        ├── Sensör   (bearing-only)          ┐
        ├── Filtre   (RP-EKF / EKF / PF)     │  pluggable yuvalar
        └── Manevra  (RH / RL / kahin)       ┘  registry ← scenarios.json
```

- **World** gerçeği tutar ve döngüyü çevirir. Gerçek → belief sınırını geçen
  tek şey **bearing**; menzil asla geçmez (problemin bilgi asimetrisi budur).
- **Platform** durum + opsiyonel sensor/filter/maneuver taşır. Hedef = boş
  bileşenli platform.
- Kahin ve GT-ödül gibi *gerçeği tüketen* parçalar, belief'ten (`ctx`) değil,
  World'ün ayrı **ayrıcalıklı kanalından** (`true_target_state()`) beslenir.

---

## 2. Bileşen sözleşmeleri (arayüzler)

Aşağıdaki imzalar v1 için kilitli. Gövdeler ayrı; bunlar sadece sözleşme.

### Sensor (ölçüm modelinin TEK kaynağı)
Ölçüm modeli bugün üç yerde ayrı yazılı (gerçek ölçüm üretimi, filtre `H`, FIM
`H`). Sensör bunların tek kaynağı olur; measure + filtre + FIM ondan beslenir.

```python
class Sensor(Protocol):
    sigma: float
    R: np.ndarray                                  # bearing-only: [[sigma**2]]
    def visible(self, own_pos, own_hdg, tgt_pos) -> bool: ...   # v1: her zaman True
    def sample(self, own_pos, tgt_state, t, rng) -> float | None: ...  # bearing; None = görüş yok
    def h(self, x, own_pos) -> float: ...          # beklenen bearing
    def H(self, x, own_pos) -> np.ndarray: ...     # 1x4 Jacobian (filtre + FIM ortak)
```

### Filter (dışa TEK (mean, cov); planlamaya ayrı pencere)
```python
class Filter(Protocol):
    def init(self, z0, own_pos): ...
    def step(self, z, own_pos): ...                # z None ise predict-only (baffle sonrası)
    @property
    def estimate(self) -> np.ndarray: ...          # raporlanan mean [x,y,vx,vy]
    @property
    def covariance(self) -> np.ndarray: ...        # raporlanan 4x4
    def propagate_state(self, x, dts) -> np.ndarray: ...   # deterministik nominal yörünge; Q yok, cov yok
    def planning_states(self): ...                 # [(state4, weight)] — FIM planlayıcı için  [AÇIK KARAR A]
```
- `propagate_state(x, dts)`: `dts` dizi (skaler gelse `atleast_1d`), dönüş `(N,4)`.
  CV için vektörize; nonlineer modelde içeride döngü, çıktı yine `(N,4)`.
- `planning_states()`: RP-EKF → `list(zip(xs, wb))`; düz EKF → `[(estimate, 1.0)]`;
  PF → alt-örneklenmiş partiküller + ağırlıkları. **Bkz. Açık karar A.**

### Maneuver
```python
class Maneuver(Protocol):
    def decide(self, ctx) -> int: ...              # aksiyon indeksi

class StochasticManeuver(Maneuver):                # RL için ek yüzler
    def action_dist(self, ctx) -> np.ndarray: ...  # softmax olasılıkları (veri toplama)
    def value(self, ctx) -> float: ...             # kritik tahmini
    def decide(self, ctx): return int(np.argmax(self.action_dist(ctx)))  # greedy varsayılan
```
- Kahin `Maneuver`'dır ama `StochasticManeuver` değildir; gerçeği `ctx`'ten değil,
  kurulumda bağlanan `truth_source`'tan alır (ayrıcalıklı kanal).

### ManeuverContext (platform kurar, dürüst — gerçek taşımaz)
```python
@dataclass
class ManeuverContext:
    own: np.ndarray            # [x,y]
    hdg: float
    own_hist: np.ndarray       # geçmiş iz (FIM'in geçmiş katkısı için)
    t: float
    estimate: np.ndarray       # filtre kestirimi (mean)
    covariance: np.ndarray     # maliyet genişlerse / RL gözlemi için
    propagate_state: callable  # filtrenin ileri-modeli (override edilebilir)
    planning_states: callable  # RH bunu kullanır  [AÇIK KARAR A]
    action_set: np.ndarray     # platform kabiliyeti (RH ve RL AYNI seti görür — adil kıyas)
```

### RewardFn (pluggable)
```python
class RewardFn(Protocol):
    def baseline(self, world): ...                 # karar aralığı BAŞINDA referans
    def __call__(self, world, prev, action) -> float: ...  # aralık SONUNDA ödül

class BeliefLogDetReward:   # v1 varsayılan — bugünkü davranış
    ...
class GTFIMReward:          # sonra — gerçeği ayrıcalıklı kanaldan tüketir
    ...
```

### Initializer (pluggable — sabit senaryo vs randomization)
```python
class Initializer(Protocol):
    def draw(self, rng) -> dict: ...               # platform state spec'leri

class FixedScenario:        # Durum 2 (kıyas/regresyon) — scenarios.json
    ...                     # her reset aynı geometri; tohum yalnızca ölçüm gürültüsü
class RandomizedScenario:   # Durum 1 (eğitim) — domain randomization
    ...
```

### Platform
```python
class Platform:
    state: np.ndarray          # [x,y,vx,vy]
    hdg: float
    hist: list                 # kendi izi (World propagate ettikçe büyür)
    sensor:   Sensor   | None
    filter:   Filter   | None
    maneuver: Maneuver | None
    turn_rate_max: float       # v1: np.inf (ani dönüş)
    action_set: np.ndarray
    def propagate(self, dt, desired_hdg=None): ...     # kinematik + dönüş kısıtı
    def maneuver_context(self, t) -> ManeuverContext: ...
    def observe(self) -> np.ndarray: ...               # = encode_obs(ctx); v1: mevcut 15-boyut
```

### World
```python
class World:
    def __init__(self, platforms, reward_fn, initializer,
                 dt, total_time, plan_every, seed): ...
    def reset(self, seed=None) -> np.ndarray: ...      # initializer.draw → platformlar → filtre init → reward baseline
    def apply(self, action) -> tuple: ...              # bir KARAR aralığı (plan_every ölçüm); (obs, reward, done, info)
    def true_target_state(self) -> np.ndarray: ...     # ayrıcalıklı kanal (kahin + GT-ödül)
```

### TMAGym (ince Gymnasium adaptörü — TMA mantığı YOK)
```python
class TMAGym(gym.Env):        # gymnasium/torch yalnızca burada (tembel import)
    # observation_space: Box, şekil = observe() uzunluğundan TÜRETİLİR (elle 15 yazma)
    # action_space: Discrete(len(action_set))
    # reset -> (obs.astype(float32), {})
    # step  -> (obs.astype(float32), float(reward), terminated=False, truncated=done, info)
```
- **`terminated=False`, `truncated=done`**: episode "hedef çözüldü" ile değil,
  `total_time` dolduğu için biter → her zaman truncation, asla terminal. Bu
  ayrım, elle-numpy GAE'deki "truncation'ı terminal sayma" bug'ının doğru
  tarafıdır; SB3 bootstrap'ı bu sayede doğru yapar.

---

## 3. Temel ilkeler

- **Gerçek/belief sınırı:** yalnızca bearing geçer. Ownship hedefin gerçek
  menzilini bilmez; sensör geçici olarak geometriyi okur, tek çıktı bearing'dir.
- **İki saat:** manevra `plan_every`×`dt` = 60 s'de bir karar verir; filtre her
  `dt` = 10 s'de bir ölçüm alıp güncellenir. Bir `apply(a)` = bir karar aralığı =
  6 ölçüm adımı. Bir RL step'i = bir karar aralığı.
- **İki sürücü, tek zemin:** Durum 1 (RL veri toplama) ve Durum 2 (kıyas) ikisi de
  `maneuver_context()` + `apply()` üzerinde çalışır. Manevra nesnesi ikisinde de
  aynı; Durum 1 ayrıca `action_dist`/`value` yüzlerini tüketir.
- **Ayrıcalıklı bilgi yalıtılmış:** kahin ve GT-ödül gerçeği yalnızca eğitim/
  değerlendirme zamanında, `ctx`'ten değil ayrı ve görünür bir kanaldan kullanır
  (asimetrik aktör-kritik).

---

## 4. Kurulum (registry + JSON)

```python
SENSORS   = {"bearing_only": BearingOnlySensor}
FILTERS   = {"rp_ekf": RPEKFBank, "ekf": EKF, "pf": ParticleFilter}
MANEUVERS = {"receding_horizon": RHManeuver, "rh2": RH2Maneuver,
             "oracle": OracleManeuver, "ppo": PPOManeuver, "straight": StraightManeuver}
REWARDS   = {"belief_logdet": BeliefLogDetReward, "gt_fim": GTFIMReward}
INITS     = {"fixed": FixedScenario, "randomized": RandomizedScenario}
```

`scenarios.json` şeması:
```json
{
  "platforms": [
    {"role": "target",
     "state": {"bearing": 45, "range": 10000, "course": 225, "speed": 5}},
    {"role": "ownship",
     "state":    {"x": 0, "y": 0, "course": 0, "speed": 8, "turn_rate_max": null},
     "sensor":   {"type": "bearing_only", "sigma_deg": 1.0},
     "filter":   {"type": "rp_ekf", "n_hyp": 6, "r_range": [3000, 18000]},
     "maneuver": {"type": "receding_horizon", "horizon": 300}}
  ],
  "reward":      {"type": "belief_logdet"},
  "initializer": {"type": "fixed"},
  "simulation":  {"dt": 10, "total_time": 1200, "plan_every": 6, "seed": 42}
}
```
`build_platform(spec)` tip string'i registry'den sınıfa çözer; `{"type": ...}`
dışındaki anahtarları kwarg olarak geçer. Kahin özel: `truth_source` JSON'dan
değil, `build_world` içinde World'den bağlanır.

---

## 5. Regresyon — altın referans

Aşağıdaki tablo (`tma_full.py`, seed 100–119, domain randomization) v1 boyunca
her adımda TUTMALIDIR. `SENARYOLAR.md` ile birebir doğrulandı.

| Politika          | medyan | ort    | >1km  | ödül |
|-------------------|--------|--------|-------|------|
| Kahin FIM         | 220 m  | 2088 m | 1/20  | 21.4 |
| Belief FIM 2B     | 297 m  | 293 m  | 0/20  | 20.6 |
| Belief FIM 1B (RH)| 195 m  | 430 m  | 2/20  | 21.3 |
| Rastgele          | 711 m  | 915 m  | 7/20  | 16.1 |
| Düz rota          | 2712 m | 4304 m | 15/20 | 12.0 |

**RNG çekme sırası kuralı (bit-birebir için — Açık karar B):**
Her episode'da rastgele sayı çekme sırası mevcut kodla AYNI kalmalı:
1. `reset`: domain randomization sırası → `R0`, `b0`, `spd`, `hdg`, sonra own `hdg`.
2. İlk ölçüm gürültüsü (`_measure` içindeki `standard_normal`).
3. Her adımda ölçüm gürültüsü.
Harness her politikanın 20-tohum sonuç vektörünün TAMAMINI (medyanı değil)
`np.array_equal` ile mevcut çıktıya karşı doğrular.

---

## 6. Uygulama sırası

Her adım yukarıdaki tabloyu birebir tutmalı. Adım 7 kasıtlı davranış değiştirir.

0. **Altın referansı çivile** — mevcut kodu koştur, 20-tohum vektörlerini kaydet. ✅ yapıldı
1. **Sensör** çıkar (bearing-only, `visible→True`); measure + filtre `H` + FIM `H` ondan.
2. **Filtre** protokolü (`RPEKFBank`'ı `estimate`/`covariance`/`propagate_state`/`planning_states`'e uydur).
3. **Platform + World** — döngüyü taşı, RNG sırasını sabitle.
4. **Manevra** — mevcut politikaları `decide(ctx)`'li nesnelere sar.
5. **RewardFn + Initializer** pluggable.
6. **TMAGym** + `check_env` (gymnasium; SB3/torch — Windows sorunu burada).
7. **LOS-göreli obs** — davranışı bilerek değiştirir; regresyonla değil "iyileştiriyor mu" kıyasıyla doğrulanır.

---

## 7. Açık kararlar (ONAY BEKLİYOR)

### A — RH planlayıcı neyin üzerinden FIM alıyor?
> ⚠️ **BU KARAR SONRADAN TERSİNE ÇEVRİLDİ — bkz. §11.A′.** Aşağıdaki metin
> tarihsel kayıt olarak duruyor; güncel davranış seçenek (ii)'dir.

Mevcut `rh_policy`, 6 hipotez üzerinden ağırlıklı logdet topluyor:
`Σ wbᵢ · log det J(xsᵢ)`. Tek moment-eşlenmiş ortalamaya çökertmek RH'nin
195 m'sini değiştirir (belief-FIM'in kalitesi bu beklentiden gelir).
- **Öneri (i):** filtre `estimate`/`covariance`'ı raporlar (tek), AYRICA
  `planning_states()` açar (ağırlıklı hipotezler); RH bunu kullanır. RP-EKF →
  `(xs, wb)` (195 m korunur), düz EKF → `[(mean, 1.0)]` (certainty-equivalent).
- Alternatif (ii): RH'yi tek ortalamaya çökert, 195 m'yi yeniden baz al.
- **Karar: (i) öneriliyor.**

### B — Bit-birebir mi, istatistiksel mi regresyon?
Temiz sayfa RNG sırasını otomatik korumaz.
- **Öneri (i):** yeni kodda çekme sırasını bilerek koru → bit-birebir güvenlik ağı.
- Alternatif (ii): istatistiksel eşdeğerlik (medyanlar tolerans içinde).
- **Karar: (i) öneriliyor** (Kahin'in tek uçuk tohumu istatistiksel bandı bozar).

---

## 8. v1 varsayılanları / kısıtları

- Gözlem: **mevcut 15-boyut mutlak çerçeve** (LOS-göreli düzeltme = Adım 7).
- Aksiyon: 12 ayrık mutlak rota (`deg2rad(arange(0,360,30))`); göreli aksiyon = sonra.
- Dönüş: **ani** (`turn_rate_max=inf`); dönüş-oranı kısıtı = sonra.
- Durum: 4-boyut CV. `propagate_state` daha yüksek boyuta açık.
- Görüş: her yönden (`visible→True`); baffle = sonra, `visible()` yerinde duruyor.
- Bağımlılık: çekirdek saf numpy. `gymnasium`/`torch` yalnızca `TMAGym` içinde
  tembel import. SB3 venv'i için Python 3.12 önerilir (3.13/torch `c10.dll` sorunu).
- Paket düzeni: küçük bir paket (`tma/` altında `sensor.py`, `filter.py`,
  `platform.py`, `world.py`, `maneuver.py`, `reward.py`, `initializer.py`,
  `registry.py`, `gym_env.py`) + `regression_test.py`.

---

## 9. Standart eşlemesi (referans)

- Problem: POMDP / sensor management / active perception; "observer trajectory
  optimization". D-optimal FIM = optimal experiment design.
- Döngü: Sense–Plan–Act.
- Kestirim/kontrol ayrımı: separation principle / certainty equivalence (GNC).
- Tak-çıkar: Strategy pattern + dependency injection; ports & adapters (hexagonal).
  World port tanımlar, somut sınıflar adaptör, TMAGym = Gym port'una adaptör.
- Öğrenme: Gymnasium API + Stable-Baselines3 (fiili standart).
- Ayrıcalıklı bilgi: asymmetric actor-critic / learning from privileged information.
- İki saat: temporal abstraction (options / frame-skip); receding horizon + iç filtre.

---

## 10. Ek kararlar (bu tur) — §5–8'i günceller

**Kararlar A ve B:** her ikisi de seçenek (i) — KABUL EDİLDİ.
- A: filtre tek `(mean, cov)` raporlar + `planning_states()` açar; RH ağırlıklı
  hipotezler üzerinden FIM alır (195 m korunur).
  **→ A sonradan tersine çevrildi, bkz. §11.A′.**
- B: yeni kodda RNG çekme sırası bilerek korunur → bit-birebir regresyon.

### C — Açılış bacağı: kerterize DİK, +90° (kahin hariç tüm manevralar)
> ℹ️ **Ofset sonradan parametrik yapıldı — bkz. §11.G′.** +90° hâlâ varsayılan; artık
> `opening_offset_deg` ile değiştirilebiliyor (0 = ilk temas kerterizine git).

Kahin dışındaki tüm manevralarda (RH, RH2, RL, rastgele, düz) ilk bacak, ölçülen
kerterize DİK (+90°) yönünde.
- **Yön:** `wrap(bearing0 + pi/2)`, en yakın ayrık rotaya (`action_set`) yuvarlanır.
  `bearing0` = açılış anındaki belief kerterizi (`ctx.estimate` yönü; t=0'da z0'a eşit).
  NOT: +90° ve −90° (`bearing0 ± pi/2`) gözlenebilirlik açısından simetriktir; ikisi de
  maksimum bearing rate verir. v1 varsayılanı +90°; istenirse iki kemer adayından
  logdet'i yüksek olanı seçilebilir.
- **Nerede yaşar:** `World.reset` içinde sabit bir ön-bacak olarak koşulur;
  `ownship.maneuver.privileged` True ise (kahin) atlanır. RL episode'u ön-bacaktan
  SONRA başlar → RL kendisinin vermediği kararı öğrenmez, maskeleme gerekmez, tüm
  manevralar aynı açılışı paylaşır (ön-bacağın bilgi kazancı hepsinde ortak, sabit
  ofset — adil kıyas).
- **GÖZLENEBİLİRLİK NOTU:** kerterize dik (kemer) hareket = LOS'a maksimum enine bileşen;
  kendi-kaynaklı bearing rate en yüksek → menzil belirsizliğini en hızlı çözen açılış.
  Bearings-only TMA'nın klasik optimal açılış manevrası budur.

### D — Sınırlı dönüş oranı
Platform hedef rotaya `turn_rate_max` (config, örn. 3 derece/s) ile döner; ani değil.
- **Platform:** `propagate(dt, desired_hdg)` her `dt`'de mevcut `hdg`'yi
  `min(turn_rate_max*dt, kalan_aci)` kadar hedefe yaklaştırıp hareket eder.
  `turn_rate_max` JSON'dan: `ownship.state.turn_rate_max` (derece/s).
- **ALT-KARAR (girdi bekliyor) — planlayıcının kendi gelecek izi:**
  - (a) *ani-dönüş yaklaşımı* (ÖNERİ, v1): `_own_future` adayları ani dönüşle skorlar;
    platform gerçekte oranıyla döner; fark 60 s'lik replan ile düzelir. Basit, hızlı,
    vektörize. Küçük planlayıcı/platform uyumsuzluğu.
  - (b) *tutarlı modelleme*: `_own_future` dönüş-oranı-sınırlı yayı simüle eder (aday
    başına küçük döngü). Daha doğru, daha yavaş, daha çok kod.
  - **Öneri: (a) v1 için**, sonra istenirse (b).

### Faz'lama — §5 ve §6'yı günceller
C ve D davranışı BİLEREK değiştirir; §5'teki altın tablo yalnızca saf refactor için
geçerli.
- **Faz 1 (Adım 1–6):** saf refactor, ani dönüş + mevcut açılış → §5 tablosunu
  BİT-BİREBİR üret. Refactor doğruluğunu kanıtlar.
- **Faz 2 (Adım 6.5):** C + D aç → YENİ referans tablosu üret; Faz 1 ile öncesi/
  sonrası kıyas. "Refactor bozmadı" ile "tasarım değişikliği ne yaptı" ayrışır.
- **Faz 3 (Adım 7):** LOS-göreli obs.

### §8 güncellemesi
- Dönüş: artık **sınırlı dönüş oranı** (ani değil) — `turn_rate_max` config.
- Açılış bacağı: kahin hariç tüm manevralarda ilk bacak = kerterize dik (+90°).

---

## 11. Karar revizyonları (sonraki turlar) — §7 ve §10'u günceller

### A′ — RH/RH2 FIM'i TEK ağırlıklı kestirim üzerinden alır (§7.A'yı tersine çevirir)

**Yeni karar:** §7.A'da reddedilen **seçenek (ii)** benimsendi. `RHManeuver` ve
`RH2Maneuver` artık FIM'i filtrenin tek, moment-eşlenmiş kestirimi (`ctx.estimate`,
ağırlık 1.0) üzerinden hesaplıyor — certainty-equivalent planlama. RP-EKF bankasının
6 hipotezi planlamaya girmiyor.

**Uygulama:** `tma/maneuver.py::_fim_targets(ctx, use_hypotheses)`. Her iki manevra da
`use_hypotheses=False` varsayılanıyla kuruluyor; `True` verilince eski (i) davranışına
döner — kıyas koşmak için bırakıldı.

**Ölçülen etki** (60 episode, `scenarios.json`, `total_time=900`, `turn_rate_deg=3`):

| RH modu | ort | medyan | std | maks |
|---|---|---|---|---|
| tek ağırlıklı kestirim (yeni, (ii)) | 224.7 | 162.5 | 276.5 | 2060.5 |
| 6 hipotez (eski, (i)) | 230.5 | 162.5 | 282.2 | 2060.5 |

Medyan ve maks birebir aynı, ortalamada ~%2.5 fark. §7.A'daki "195 m'yi değiştirir"
uyarısı bu senaryo/parametre setinde pratikte gerçekleşmedi — filtre erken yakınsadığı
için hipotez bankası tek moda çöküyor. Farkın anlamlı olabileceği yer menzil
belirsizliğinin yüksek kaldığı erken/zayıf-gözlenebilir geometriler.

**Not:** `planning_states()` filtre sözleşmesinde KALIYOR (§2) — yalnızca RH/RH2
varsayılan olarak kullanmıyor. Sözleşme kaldırılmadı ki (i)'ye dönüş tek bayrakla
mümkün olsun.

### B′ — PTB kuralı yeniden tanımlandı + sürekli açı (§8'deki "12 ayrık rota"yı deler)

**Yeni kural** (`PTBManeuver.decide`): kerterize göre `±offset_deg` konumundaki iki
adaydan, mevcut rotaya (`ctx.hdg`) açısal olarak **UZAK** olanı seç. Tek kural budur;
hangi yönden dönüleceği manevranın işi değil (bkz. C′).

**Sürekli açı:** `Maneuver.continuous` bayrağı eklendi. `False` (varsayılan, tüm diğer
manevralar + RL) → `decide()` `action_set` indeksi döndürür, `World.apply()` `actions[i]`
ile açıya çevirir. `True` (PTB) → doğrudan radyan açı; ayrık ızgaraya yuvarlama yok.

*Gerekçe:* 30°'lik ızgarada PTB'nin ideal açısı ardışık kararlarda ~12° kaysa bile aynı
kovada kalıp üst üste **aynı** aksiyonu üretiyordu (gözlemlenen artefakt). Yuvarlama
PTB'yi yapay olarak sakatlıyordu.

*Kıyas adaleti uyarısı:* RL 12 ayrık rotayla sınırlıyken sürekli açı kullanan manevra
daha ince nişan alabilir — §2'deki "RH ve RL AYNI seti görür" ilkesiyle gerilim var.
Bayrak bilinçli seçilmeli; not `Maneuver.continuous` tanımının yanında.

**Parametrik ofset:** `PTBManeuver(offset_deg=90.0)`. 90 dışı ofsetlerde iki aday artık
180° aralıklı değildir; "uzak olanı seç" kuralı yine geçerli. `analyze.py`'deki
`PTB_OFFSETS_DEG` listesi birden çok varyantı aynı koşuda yan yana kıyaslar.

**Dokunulmayan:** makalenin açılış bacağındaki `ptb_heading()` kasıtlı olarak eski
hâlinde bırakıldı (makale reprodüksiyonunu bozmamak için). Yani `paper_repro.py`
etkilenmedi; bu kural yalnızca genel dünyadaki `PTBManeuver`'a ait.

### C′ — Dönüş YÖNÜ platformun kararı, kerteriz-farkındalıklı (Karar D'yi genişletir)

**Sorumluluk ayrımı:** manevra HANGİ açıya dönüleceğini, platform o açıya HANGİ
taraftan (sancak/iskele) dönüleceğini belirler.

**Yeni davranış** (`Platform.propagate`): en kısa yol varsayılan; ancak en kısa yol
`kerteriz + 180°` (hedefe sırt) açısının üzerinden geçiyorsa **uzun yoldan** dönülür.
Kerteriz dışarıdan parametre olarak GELMEZ — platform kendi sensöründen okur
(`self.sensor.last_z`).

**Gereken altyapı:** `BearingOnlySensor` artık son ham ölçümü `self.last_z`'de saklıyor.
RNG çekme sırası korundu (tek `standard_normal()` çağrısı) — §5'teki bit-birebir kuralı
bu değişiklikten etkilenmez.

**Kapsam:** TÜM manevralar için geçerli (genel platform davranışı), yalnızca PTB için
değil. `turn_rate_max=inf` (ani dönüş) durumunda mantık devre dışı — dönüş anlık
olduğu için "yol üzerinde" bir açı yoktur.

**Regresyon etkisi:** sonlu dönüş oranı kullanan tüm koşularda dönüş yönü değişebilir →
`golden.npz` bit-birebir testi etkilenir. (Bu test zaten `PLAN_EVERY`/`T_TOTAL`
değişiklikleriyle geçmiyordu.)

### D′ — Senaryo dosyası: `own_speed` artık okunuyor (§4 şeması ile kod arasındaki boşluk)

`scenarios.json`'daki `ownship.speed` alanı **sessizce yok sayılıyordu** — gözlemci hızı
her zaman `world.py::OWN_SPD`'den geliyordu (JSON'a 99 yazılsa bile 8.0 kalıyordu).

**Düzeltme:** `FixedScenario.draw()` artık `own_speed` döndürüyor, `World.reset()`
`None` değilse uyguluyor. Geriye dönük uyumlu: `RandomizedScenario` ve
`PaperInitializer` bu alanı döndürmediği için platformun kurulum hızı korunur.

**Kalan boşluk:** §4 şemasında belgelenen `ownship.state.turn_rate_max` alanı hâlâ
okunmuyor; dönüş oranı `build_world(turn_rate_deg=...)` üzerinden geliyor.

### E′ — `total_time` / `plan_every` artık `build_world` parametresi

Önceden yalnızca `world.py`'nin modül sabitleri (`T_TOTAL`, `PLAN_EVERY`) üzerinden
ayarlanabiliyordu. Artık `build_world(..., total_time=None, plan_every=None)`;
`None` verilirse modül sabitine düşer. `make_tma_gym`, `tma/rollout.py`, `main.py`
ve `analyze.py` (`--total-time`) bu parametreyi uçtan uca geçiriyor.

*Gerekçe (yaşanmış hata):* `T_TOTAL` oturum ortasında 1200→900 değiştirildi; ajan
1200'lük ufukla eğitilmişken 900'lük ufukta değerlendirildi ve bu fark uzun süre
fark edilmedi. Ölçülen sonuç — ajan **+332%** kötüleşti (196.5 → 849.4 m), klasik
planlayıcılar ise yalnızca +11…60%. Sebep: episode başına karar sayısının değişmesi
(20→15) VE gözlemdeki `world.t / world.total_time` özelliğinin ölçeğinin kayması
(ajan eğitimde hiç görmediği değerlerle karşılaşıyor). Ufkun çağrı yerinde açıkça
belirtilmesi bu sınıf hataları görünür kılar.

### F′ — Tekrarlanabilirlik: torch tek iş parçacığı

SB3/torch varsayılan çok-iş-parçacıklı BLAS ile çalışırken kayan nokta toplama sırası
process'ten process'e değişiyor. Bu sistem neredeyse-eşit skorlu `argmax` kararlarına
karşı **kelebek etkisi** gösterdiği için aynı tohum farklı koşularda farklı sonuç
veriyordu (aynı seed'de ort. `pos_err` 7.3 → 59.8 → 333.1 gibi). `analyze.py`
başlangıçta `torch.set_num_threads(1)` çağırır; karşılaştırma koşan her yeni script
aynısını yapmalı.

### G′ — Açılış bacağı ofseti parametrik: "ilk temas kerterizine git" eklendi (§10.C'yi genişletir)

**Yeni durum:** açılış bacağının kerterize ofseti artık çağrı yerinden seçiliyor.
`World.opening_offset` (radyan) baştan beri vardı ama üstündeki hiçbir katman geçirmiyordu —
ölü bir düğmeydi. Zincir uçtan uca bağlandı ve derece cinsinden bir sınır parametresi eklendi
(`turn_rate_deg` ile aynı birim konvansiyonu):

```
World(opening_offset=rad)  ←  build_world(opening_offset_deg=)  ←  make_tma_gym(opening_offset_deg=)
                                                                ←  rollout.run_*_episode(opening_offset_deg=)
                                                                ←  main.py / analyze.py  --opening-offset-deg
```

- `90` (varsayılan): kerterize DİK — Karar C, davranış değişmedi. `np.deg2rad(90.0)` bit
  olarak `np.pi/2`'ye eşit olduğu için varsayılan yol birebir korundu.
- `0`: İLK TEMAS KERTERİZİNE GİT — hareket LOS boyunca, kendi hareketinin ürettiği enine
  bileşen ~0.
- Ara değerler serbest (`45`, `270`, …); `build_world` ikisi birden verilirse dereceyi
  kazandırır.

**Ölçülen etki** (60 episode, `scenarios.json`, `total_time=900`, `turn_rate_deg=3`):

| Manevra | ofset | ort | medyan | maks | >1km | son menzil | hata/menzil | along_std/menzil |
|---|---|---|---|---|---|---|---|---|
| Düz | 90° | 1013.4 | 957.9 | 3217.0 | 28/60 | 11905 m | 8.45% | 31.5% |
| Düz | 0° | 202.7 | 195.1 | 620.7 | 0/60 | 2920 m | 10.22% | 31.3% |
| RH | 90° | 168.9 | 132.2 | 674.0 | 0/60 | 5322 m | 3.14% | 5.8% |
| RH | 0° | 174.7 | 112.6 | 607.2 | 0/60 | 3260 m | 5.20% | 9.5% |

**Yorum — mutlak `pos_err` bu kıyasta yanıltıyor.** Düz rotada ofset 0 mutlak hatayı 5 kat
iyileştiriyor gibi görünüyor, ama bu bir gözlenebilirlik kazancı DEĞİL: konum hatası ≈
menzil × kerteriz hatası olduğu için menzili 11.9 km'den 2.9 km'ye kapatmak hatayı
mekanik olarak küçültüyor. Ölçekten arındırılmış metrikler bunu ele veriyor — `along_std`
(LOS boyunca = menzil belirsizliği) menzile oranla **%31.5 → %31.3**, yani pratikte
değişmiyor: menzil hiç çözülmemiş, yalnızca yakınlaşılmış. Aynı yerde `hata/menzil` bile
kötüleşiyor (%8.45 → %10.22).

RH'de tablo tersine dönüyor: mutlak ortalama neredeyse eşit (168.9 vs 174.7) ama ölçekten
arındırılmış `along_std/menzil` **%5.8 → %9.5**, `hata/menzil` **%3.14 → %5.20** — dik açılış
menzili gerçekten daha iyi çözüyor. Karar C'nin gözlenebilirlik gerekçesi ayakta.

**Kural:** açılış bacağı varyantlarını mutlak `pos_err` ile kıyaslama. `along_std/menzil`
veya `hata/menzil` kullan; aksi halde menzil kapatma artifaktını gözlenebilirlik kazancı
sanırsın.

**Kapsam ve uyarılar:**
- Açılış bacağı World seviyesinde ve Kahin hariç TÜM manevralarda ortak, yani bu bir
  *koşu-seviyesi* ayar — iki ofseti kıyaslamak için analizi iki kez koştur.
- Varsayılan `PLAN_EVERY=30` ile açılış bacağı 900 s'lik episode'un üçte birini yiyor;
  ofsetin ağırlığı bu yüzden yüksek.
- RL tarafında ofset ajanın gördüğü BAŞLANGIÇ DURUMUNU değiştirir. Eğitim ve değerlendirmede
  aynı değeri kullan — E′'deki ufuk uyumsuzluğuyla aynı sınıf hata.
- Regresyon: `tests/test_phase2.py` içinde dört test (varsayılan dik açılış, ofset 0,
  derece/radyan denkliği, ofsetin sonuca bağlı olduğu).
