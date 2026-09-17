"""Koşum konfigürasyonu — eğitimde kaydedilir, değerlendirmede beyan edilir.

main.py eğitim bitince modelin yanına bir sidecar JSON yazar
(`models/x.zip` -> `models/x.run.json`). analyze.py onu okur, AYARLAR'daki
OVERRIDES'ı üzerine uygular ve aradaki FARKI başlıkta bastırır.

Amaç kaymayı ENGELLEMEK değil: ajanı eğitilmediği koşulda denemek kasıtlı bir
deneydir (dağılım kayması / genelleme testi). Amaç kaymanın ÖRTÜK olmaktan
çıkması — hangi düğmenin oynadığı çıktıda yazsın, kaza ile kasıt ayrışsın.

Not: klasik planlayıcılar (RH/Kahin/PTB) her kararda mevcut belief'ten yeniden
planladığı için kaymadan etkilenmez; kıyasın kontrol grubu onlardır.
"""
import json
import os

# Dünyayı şekillendiren anahtarlar — buradaki fark ajanı eğitilmediği koşula sokar.
WORLD_KEYS = ("scenario", "randomized", "obs", "opening_leg",
              "opening_offset_deg", "turn_rate_deg", "total_time")

# Yalnızca kayıt amaçlı: değerlendirmeyi etkilemez, modeli tanımlar.
TRAIN_KEYS = ("algo", "timesteps", "seed")


def sidecar_path(model_path):
    """models/x veya models/x.zip|.npz -> models/x.run.json"""
    base = model_path
    for ext in (".zip", ".npz"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    return base + ".run.json"


def save(model_path, cfg):
    """Eğitim ayarlarını modelin yanına yazar; yazılan yolu döner."""
    path = sidecar_path(model_path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: cfg.get(k) for k in WORLD_KEYS + TRAIN_KEYS}, f,
                  indent=2, ensure_ascii=False)
    return path


def load(model_path):
    """Sidecar varsa dict, yoksa None (eski modeller için)."""
    path = sidecar_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _same(a, b):
    """3.0 ile 3 farkı kaymadır demesin; None güvenli."""
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def resolve(train_cfg, overrides):
    """(eval_cfg, shifts) döner.

    shifts: [(anahtar, eğitim değeri, değerlendirme değeri)] — yalnızca WORLD_KEYS.
    """
    eval_cfg = dict(train_cfg)
    eval_cfg.update({k: v for k, v in overrides.items() if k in WORLD_KEYS})
    shifts = [(k, train_cfg.get(k), eval_cfg.get(k))
              for k in WORLD_KEYS if not _same(train_cfg.get(k), eval_cfg.get(k))]
    return eval_cfg, shifts


def _fmt(v):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "E" if v else "H"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def banner(model_path, train_cfg, eval_cfg, shifts, source):
    """Eğitim vs değerlendirme koşullarını ve aradaki kaymayı bastırır."""
    line = lambda cfg: "  ".join(f"{k}={_fmt(cfg.get(k))}" for k in WORLD_KEYS)
    print(f"\n[config] model = {model_path}")
    print(f"         eğitim ayarı kaynağı: {source}")
    print(f"         eğitim        : {line(train_cfg)}")
    print(f"         değerlendirme : {line(eval_cfg)}")
    if shifts:
        print("         KAYMA (kasıtlı OOD testi): "
              + ", ".join(f"{k} {_fmt(a)} -> {_fmt(b)}" for k, a, b in shifts))
        if any(k == "total_time" for k, _, _ in shifts):
            print("         UYARI: total_time gözlemdeki t/total_time özelliğini de yeniden"
                  " ölçekler (bkz. tma/obs.py) — karar sayısından bağımsız ek bir kaymadır.")
    else:
        print("         KAYMA yok — ajan eğitim koşullarında değerlendiriliyor.")
    print()
