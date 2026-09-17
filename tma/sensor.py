"""Bearing-only sensör — ölçüm modelinin TEK kaynağı.

Bir platforma takılır; başka bir platformun (hedefin) o anki konumuna göre gürültülü
kerteriz (bearing) üretir. Ölçüm modeli (h, H, R) hem filtre hem FIM planlayıcı
tarafından ortak kullanılır — böylece model tek yerde tanımlıdır (eskiden _measure,
_bank_update ve _fim_logdet'te üç ayrı kopyaydı).

Kerteriz konvansiyonu (mevcut kodla birebir):
    beta = atan2(dx, dy),  dx = doğu bileşeni, dy = kuzey bileşeni
yani kuzey referanslı, saat yönünde ölçülen kerteriz.
"""
import numpy as np


class BearingOnlySensor:
    """Pasif kerteriz sensörü. v1: her yönden temas (baffle yok)."""

    privileged = False   # gerçeği kullanmaz; belief üzerinden çalışır

    def __init__(self, sigma_deg=1.0):
        self.sigma = np.deg2rad(sigma_deg)        # ölçüm gürültüsü std (radyan)
        self.R = np.array([[self.sigma ** 2]])    # 1x1 ölçüm kovaryansı
        self.last_z = None                        # son ham ölçüm (platform dönüş yönü için)

    # ---- görünürlük (v1: her zaman görür; baffle sonra buraya) ----
    def visible(self, own_pos, own_hdg, tgt_pos):
        return True

    # ---- gerçek ölçüm (truth tarafı) ----
    def sample(self, own_pos, tgt_state, t, rng):
        """Gürültülü kerteriz. Görüş yoksa None.

        tgt_state: hedefin [x,y,vx,vy] durumu; konum t anında CV ile ileri taşınır
        (mevcut _measure ile birebir: kapalı-form tgt[:2] + tgt[2:]*t).
        rng çekme sırası mevcut _measure ile birebir: tek skaler standard_normal().
        """
        tgt_pos = tgt_state[:2] + tgt_state[2:] * t
        if not self.visible(own_pos, None, tgt_pos):
            self.last_z = None
            return None
        d = tgt_pos - own_pos
        self.last_z = np.arctan2(d[0], d[1]) + self.sigma * rng.standard_normal()
        return self.last_z

    # ---- ölçüm modeli (filtre + FIM ortak kullanır) ----
    def h(self, x, own_pos):
        """Beklenen kerteriz: x durumunun konumundan tahmini ölçüm."""
        dx, dy = x[0] - own_pos[0], x[1] - own_pos[1]
        return np.arctan2(dx, dy)

    def H(self, x, own_pos):
        """Anlık ölçüm Jacobian'ı (1x4): d beta / d[x, y, vx, vy].

        Konum kısmı [dy/r^2, -dx/r^2]; anlık ölçüm hıza bağlı değil -> son iki bileşen 0.
        FIM planlayıcı bu konum Jacobian'ını hareket modeliyle (CV geçişinin t
        katsayıları) kompoze ederek [c, s, t*c, t*s] satırını kurar — o adım
        planlayıcıya ait, sensöre değil.
        """
        dx, dy = x[0] - own_pos[0], x[1] - own_pos[1]
        r2 = dx * dx + dy * dy
        return np.array([dy / r2, -dx / r2, 0.0, 0.0])
