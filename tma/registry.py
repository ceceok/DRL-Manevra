"""Registry + kurulum yardımcıları.

Tip string'leri -> sınıflar. build_world tüm bileşenleri bağlar; parametrelerle
Faz 1 (varsayılan), Faz 2 (opening_leg + turn_rate_deg) ve Faz 3 (obs='los')
erişilebilir. scenarios.json için build_world_from_json.
"""
import json
import numpy as np

from .world import World, Platform, DT, OWN_SPD, SIGMA_DEG, N_HYP
from .sensor import BearingOnlySensor
from .filter import RPEKFBankFilter
from .reward import BeliefLogDetReward
from .initializer import RandomizedScenario, FixedScenario
from .obs import AbsoluteObs, LOSRelativeObs
from .maneuver import (OracleManeuver, RHManeuver, RH2Maneuver,
                       RandomManeuver, StraightManeuver)

SENSORS = {"bearing_only": BearingOnlySensor}
FILTERS = {"rp_ekf": RPEKFBankFilter}
MANEUVERS = {"oracle": OracleManeuver, "rh": RHManeuver, "rh2": RH2Maneuver,
             "random": RandomManeuver, "straight": StraightManeuver}
REWARDS = {"belief_logdet": BeliefLogDetReward}
INITS = {"randomized": RandomizedScenario, "fixed": FixedScenario}
OBS = {"absolute": AbsoluteObs, "los": LOSRelativeObs}


def build_world(seed=0, maneuver=None, reward="belief_logdet", initializer=None,
                obs="absolute", opening_leg=False, opening_offset=np.pi / 2,
                opening_offset_deg=None, turn_rate_deg=None, n_hyp=N_HYP,
                total_time=None, plan_every=None):
    """Faz 1 varsayılanları; opening_leg/turn_rate_deg ile Faz 2, obs='los' ile Faz 3.
    total_time/plan_every verilmezse World'ün (tma/world.py) modül sabitleri kullanılır.

    Açılış bacağının kerterize göre ofseti iki yoldan verilebilir: opening_offset
    (radyan, World'ün kendi birimi) veya opening_offset_deg (derece, turn_rate_deg ile
    aynı sınır birimi). İkisi birden verilirse DERECE kazanır.
        90  -> kerterize dik (varsayılan, Karar C)
        0   -> ilk temas kerterizine git (zayıf-gözlenebilirlik kıyas tabanı)"""
    if opening_offset_deg is not None:
        opening_offset = np.deg2rad(opening_offset_deg)
    sensor = BearingOnlySensor(sigma_deg=SIGMA_DEG)
    filt = RPEKFBankFilter(sensor, dt=DT, n_hyp=n_hyp)
    trm = np.inf if turn_rate_deg is None else np.deg2rad(turn_rate_deg)
    ownship = Platform(pos=np.zeros(2), hdg=0.0, speed=OWN_SPD, turn_rate_max=trm,
                       sensor=sensor, filter=filt, maneuver=maneuver)
    target = Platform(pos=np.zeros(2), vel=np.zeros(2))
    reward_fn = REWARDS[reward]() if isinstance(reward, str) else reward
    init = initializer if initializer is not None else RandomizedScenario()
    obs_enc = OBS[obs]() if isinstance(obs, str) else obs
    overrides = {}
    if total_time is not None:
        overrides["total_time"] = total_time
    if plan_every is not None:
        overrides["plan_every"] = plan_every
    return World(ownship, target, reward_fn=reward_fn, initializer=init,
                 obs_encoder=obs_enc, seed=seed, opening_leg=opening_leg,
                 opening_offset=opening_offset, **overrides)


def build_world_from_json(path, maneuver=None, seed=0, obs="absolute",
                          opening_leg=False, turn_rate_deg=None,
                          opening_offset_deg=None):
    """scenarios.json'dan FixedScenario ile tek-senaryo World kurar (Durum 2)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    init = FixedScenario(cfg)
    return build_world(seed=seed, maneuver=maneuver, initializer=init, obs=obs,
                       opening_leg=opening_leg, turn_rate_deg=turn_rate_deg,
                       opening_offset_deg=opening_offset_deg)


# ---- makale (Ristic & Arulampalam) konfigürasyonu ----
from .filter_ckf import CubatureKF
from .reward import ParetoTerminalReward
from .obs import Paper12Obs
from .maneuver import PTBManeuver, PaperPTBManeuver, ITOManeuver

FILTERS["ckf"] = CubatureKF
MANEUVERS["ptb"] = PTBManeuver              # tez kuralı (MIMARI_KARARLAR §11.B′)
MANEUVERS["paper_ptb"] = PaperPTBManeuver   # makale kuralı (Ristic & Arulampalam)
MANEUVERS["ito"] = ITOManeuver
REWARDS["pareto"] = ParetoTerminalReward
OBS["paper12"] = Paper12Obs

# NOT: makale dünyası paper_repro.build_paper_world'de (kanonik).
