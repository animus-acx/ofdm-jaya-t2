"""Utilidades del protocolo experimental común: corrida individual, estadística y tablas."""
import csv
import os
import time

import numpy as np

from ofdm import Objetivo, PRESUPUESTO, G, P_TOTAL, SIGMA2

CARPETA = "resultados"
os.makedirs(CARPETA, exist_ok=True)


def correr(alg, semilla, g=G, P=P_TOTAL, sigma2=SIGMA2, n_pob=30,
           presupuesto=PRESUPUESTO, guardar_curva=True):
    """Una corrida con presupuesto de 3000 evaluaciones contadas dentro de la función objetivo.
    La curva (mejor f hasta cada evaluación) sale del historial del contador."""
    g = np.asarray(g, float)
    dim = len(g)
    obj = Objetivo(g, P, sigma2, presupuesto)
    t0 = time.perf_counter()
    x, fx, _ = alg(obj, np.zeros(dim), np.ones(dim), dim, n_pob, presupuesto // n_pob, semilla)
    dt = time.perf_counter() - t0
    r = {"x": x, "f": fx, "t": dt, "n_eval": obj.n_eval}
    if guardar_curva:
        r["curva"] = np.minimum.accumulate(np.array(obj.historial))
    return r


def estadisticas(f):
    """Mejor, mediana, media, desviación estándar muestral (ddof = 1) y peor de f."""
    f = np.asarray(f, float)
    return {"mejor": float(f.min()), "mediana": float(np.median(f)), "media": float(f.mean()),
            "desv_std": float(f.std(ddof=1)), "peor": float(f.max())}


def fmt(v, dec=4):
    """Notación científica para valores muy pequeños; decimales fijos en el resto."""
    v = float(v)
    return f"{v:.3e}" if (v != 0 and abs(v) < 1e-3) else f"{v:.{dec}f}"


def guardar_tabla(nombre, filas, encabezado):
    with open(os.path.join(CARPETA, nombre), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(encabezado)
        w.writerows(filas)


def canal_rayleigh(dim, semilla):
    """Ganancias de potencia g_i ~ Exponencial(1) (desvanecimiento Rayleigh)."""
    return np.random.default_rng(semilla).exponential(1.0, dim)
