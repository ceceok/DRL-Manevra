"""Pluggable senaryo başlatıcılar.

draw(rng) -> {"target_state":[x,y,vx,vy], "own_pos":[x,y], "own_hdg":rad,
              "own_speed": float|None}   # own_speed opsiyonel; None ise platformunki korunur

RandomizedScenario (Durum 1 / eğitim): domain randomization. Çekme sırası
tma_full.reset ile BIREBIR: R0, b0, spd, hdg(hedef), own_hdg. İlk ölçüm gürültüsü
(bir sonraki çekim) World'de yapılır — sıra korunur.

FixedScenario (Durum 2 / kıyas): scenarios.json'dan sabit geometri; rng çekmez.
"""
import json
import numpy as np


class RandomizedScenario:
    def __init__(self, r_range=(5000.0, 15000.0), spd_range=(2.0, 8.0)):
        self.r_range = r_range
        self.spd_range = spd_range

    def draw(self, rng):
        R0 = rng.uniform(self.r_range[0], self.r_range[1])
        b0 = rng.uniform(0, 2 * np.pi)
        spd = rng.uniform(self.spd_range[0], self.spd_range[1])
        hdg = rng.uniform(0, 2 * np.pi)
        tgt = np.array([R0 * np.sin(b0), R0 * np.cos(b0),
                        spd * np.sin(hdg), spd * np.cos(hdg)])
        own_hdg = rng.uniform(0, 2 * np.pi)
        return {"target_state": tgt, "own_pos": np.zeros(2), "own_hdg": own_hdg}


class FixedScenario:
    """spec: {"target": {bearing,range,course,speed}, "ownship": {x,y,course,speed}}
    (açılar derece, kuzey referanslı)."""

    def __init__(self, spec):
        self.spec = spec

    def draw(self, rng):
        t = self.spec["target"]
        o = self.spec["ownship"]
        ox, oy = float(o.get("x", 0.0)), float(o.get("y", 0.0))
        b = np.deg2rad(t["bearing"])
        crs = np.deg2rad(t["course"])
        tgt = np.array([ox + t["range"] * np.sin(b), oy + t["range"] * np.cos(b),
                        t["speed"] * np.sin(crs), t["speed"] * np.cos(crs)])
        own_hdg = np.deg2rad(o.get("course", 0.0))
        return {"target_state": tgt, "own_pos": np.array([ox, oy], float),
                "own_hdg": own_hdg, "own_speed": o.get("speed")}

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


class PaperInitializer:
    """Ristic 2026: gözlemci orijinde; hedef d~U[dmin,dmax] mesafede, rastgele
    kerteriz; hedef rotası rastgele; hedef ve gözlemci hızı eşit (=speed)."""

    def __init__(self, dmin=18.0, dmax=26.0, speed=1.0):
        self.dmin, self.dmax, self.speed = dmin, dmax, speed

    def draw(self, rng):
        d = rng.uniform(self.dmin, self.dmax)
        brg = rng.uniform(0, 2*np.pi)
        thd = rng.uniform(0, 2*np.pi)
        tgt = np.array([d*np.sin(brg), d*np.cos(brg),
                        self.speed*np.sin(thd), self.speed*np.cos(thd)])
        return {"target_state": tgt, "own_pos": np.zeros(2), "own_hdg": 0.0}
