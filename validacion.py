"""Validación (niveles V1 a V3) de las soluciones de JAYA frente al water-filling exacto."""
import numpy as np

from ofdm import (G, P_TOTAL, SIGMA2, reparar, capacidad, water_filling, asignacion_uniforme)
from jaya import jaya
from protocolo import correr

UMBRAL_OFF = 1e-3          # una subportadora se considera apagada si p_i < 1e-3 W


def comparar_con_wf(xs, g=G, P=P_TOTAL, sigma2=SIGMA2):
    """V1: compara cada solución (fila de xs) con el water-filling exacto."""
    p_star, mu, c_star = water_filling(g, P, sigma2)
    ps = reparar(xs, P)
    c = capacidad(ps, g, sigma2)
    off_ref = p_star <= 0
    off_j = ps < UMBRAL_OFF
    return {"p_star": p_star, "mu": mu, "C_star": c_star, "p": ps, "C": c,
            "gap": c_star - c, "gap_pct": 100 * (c_star - c) / c_star,
            "dist": np.linalg.norm(ps - p_star, axis=1),
            "off_ref": off_ref,
            "off_exacto": np.all(off_j == off_ref, axis=1),
            "n_off_ok": (off_j == off_ref).sum(axis=1)}


def kkt(p, g=G, sigma2=SIGMA2):
    """V2: condiciones de optimalidad sobre una asignación p.
    Activas: nivel w_i = p_i + sigma2/g_i ~ constante (= mu). Apagadas: sigma2/g_i >= mu."""
    ruido = sigma2 / np.asarray(g, float)
    activas = p >= UMBRAL_OFF
    w = p + ruido
    mu_hat = float(w[activas].mean())
    return {"mu_hat": mu_hat, "disp_std": float(w[activas].std()),
            "rango": float(w[activas].max() - w[activas].min()),
            "n_activas": int(activas.sum()),
            "apagadas_ok": int((ruido[~activas] >= mu_hat).sum()), "n_apagadas": int((~activas).sum()),
            "w": w, "ruido": ruido, "activas": activas}


def evaluar_canal(g, P=P_TOTAL, sigma2=SIGMA2, semillas=range(1, 31), n_pob=30):
    """Corre JAYA (presupuesto 3000) sobre un canal y resume la brecha respecto del water-filling."""
    xs = np.array([correr(jaya, s, g, P, sigma2, n_pob, guardar_curva=False)["x"] for s in semillas])
    r = comparar_con_wf(xs, g, P, sigma2)
    p_u, c_u = asignacion_uniforme(g, P, sigma2)
    r.update({"C_unif": c_u, "p_unif": p_u,
              "dist_unif": float(np.linalg.norm(r["p_star"] - p_u)),
              "gan_unif_pct": 100 * (r["C_star"] - c_u) / c_u,
              "n_off_star": int(r["off_ref"].sum())})
    return r
