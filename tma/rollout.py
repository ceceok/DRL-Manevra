"""Tek-episode koşum yardımcıları — analyze.py (Monte Carlo + tek koşum) kullanır.

scenarios.json (FixedScenario) üzerinde bir episode sonuna kadar koşar. Çağıran
taraf episode SONUNDA ne çekeceğine kendi karar verir: pos_err skaleri (istatistik)
veya tam trajectory + karar izi (görselleştirme).
"""
import numpy as np

from .registry import build_world, MANEUVERS
from .initializer import FixedScenario
from .gym_env import make_tma_gym


def _decision_row(world, ctx, a):
    """Karar anındaki durum: konum, kerteriz (belief), ham ölçüm, rota, seçilen aksiyon.

    bearing_deg : filtre kestirimine göre kerteriz (manevraların gördüğü)
    z_deg       : sensörün son HAM ölçümü (platform dönüş yönünü buna göre seçer;
                  görüş yoksa None)
    """
    rel = ctx.estimate[:2] - ctx.own
    z = world.ownship.sensor.last_z if world.ownship.sensor is not None else None
    m = world.ownship.maneuver
    # continuous manevrada a zaten radyan açı; değilse action_set indeksi (bkz. Maneuver.continuous)
    hdg_cmd = a if (m is not None and m.continuous) else ctx.action_set[a]
    return {"t": world.t, "own": np.asarray(ctx.own, float).copy(),
           "bearing_deg": np.degrees(np.arctan2(rel[0], rel[1])),
           "z_deg": None if z is None else np.degrees(z),
           "hdg_deg": np.degrees(ctx.hdg), "action_deg": np.degrees(hdg_cmd)}


def run_baseline_episode(maneuver, scenario_path, opening_leg, turn_rate_deg, seed,
                         total_time=None, opening_offset_deg=90.0):
    """Klasik planlayıcıyla bir episode koşar; (world, son info) döner.

    maneuver: registry anahtarı (str) veya parametreli varyantlar için fabrika
              (çağrılabilir), örn. lambda: PTBManeuver(offset_deg=80).
    opening_offset_deg: açılış bacağının kerterize ofseti (90 dik, 0 ilk temasa git).
    info["trace"]: her karar anındaki (t, kerteriz, kendi rota, seçilen aksiyon) listesi.
    """
    man = maneuver() if callable(maneuver) else MANEUVERS[maneuver]()
    world = build_world(seed=seed, maneuver=man,
                        initializer=FixedScenario.from_json(scenario_path),
                        opening_leg=opening_leg, turn_rate_deg=turn_rate_deg,
                        opening_offset_deg=opening_offset_deg,
                        total_time=total_time)
    world.reset(seed=seed)
    done, info, trace = False, {}, []
    while not done:
        ctx = world.maneuver_context()
        a = man.decide(ctx)
        trace.append(_decision_row(world, ctx, a))
        _, _, done, info = world.apply(a)
    info["trace"] = trace
    return world, info


def run_agent_episode(model, scenario_path, obs, opening_leg, turn_rate_deg, seed,
                      total_time=None, opening_offset_deg=90.0):
    """SB3 modeliyle bir episode koşar; (env.world, son info) döner.

    opening_offset_deg EĞITIMDEKI ile aynı olmalı — açılış bacağı ajanın gördüğü
    başlangıç durumunu belirler (bkz. MIMARI_KARARLAR.md §11.E′ ufuk uyumsuzluğu).
    info["trace"]: her karar anındaki (t, kerteriz, kendi rota, seçilen aksiyon) listesi."""
    env = make_tma_gym(seed=seed, obs=obs, opening_leg=opening_leg,
                       turn_rate_deg=turn_rate_deg,
                       opening_offset_deg=opening_offset_deg,
                       initializer=FixedScenario.from_json(scenario_path),
                       total_time=total_time)
    o, _ = env.reset(seed=seed)
    done, info, trace = False, {}, []
    while not done:
        a, _ = model.predict(o, deterministic=True)
        ctx = env.world.maneuver_context()
        trace.append(_decision_row(env.world, ctx, int(a)))
        o, r, terminated, done, info = env.step(int(a))
    info["trace"] = trace
    return env.world, info
