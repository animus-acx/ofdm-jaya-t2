"""Modelo del problema T2: asignación de potencia en subportadoras OFDM.

Capacidad:  C(p) = sum_i log2(1 + g_i * p_i / sigma2)        [bit/s/Hz]
Variables:  x en [0, 1]^D  ->  operador de reparación  p_i = P * x_i / sum_j x_j
            (si sum_j x_j = 0, potencia uniforme), de modo que sum p_i = P y p_i >= 0.
Objetivo:   f(x) = - C(p(x))      (se minimiza)
"""
import numpy as np

# Datos del enunciado: 16 subportadoras, P = 1 W, sigma2 = 0.1 W
G = np.array([0.149, 1.266, 0.429, 0.686, 0.978, 1.068, 1.285, 0.019,
              0.912, 0.635, 1.689, 2.782, 1.373, 0.448, 0.167, 0.549])
P_TOTAL = 1.0
SIGMA2 = 0.1
D = 16
PRESUPUESTO = 3000


def reparar(X, P=P_TOTAL):
    """Operador de reparación vectorizado: x -> p con sum(p) = P."""
    X = np.atleast_2d(X)
    s = X.sum(axis=1, keepdims=True)
    uniforme = np.full(X.shape, 1.0 / X.shape[1])
    frac = np.where(s > 0, X / np.where(s > 0, s, 1.0), uniforme)
    return P * frac


def capacidad(p, g=G, sigma2=SIGMA2):
    """C(p) en bit/s/Hz para cada fila de p."""
    p = np.atleast_2d(p)
    return np.sum(np.log2(1.0 + g * p / sigma2), axis=1)


class Objetivo:
    """f(x) = -C(p(x)) con contador de evaluaciones DENTRO de la función objetivo.

    Acepta un vector (devuelve un escalar) o una matriz n x D (devuelve n valores).
    Guarda el valor de cada evaluación en `historial` (curva de convergencia frente
    al número de evaluaciones) y falla si se supera el presupuesto.
    """
    def __init__(self, g=G, P=P_TOTAL, sigma2=SIGMA2, presupuesto=PRESUPUESTO):
        self.g, self.P, self.sigma2, self.presupuesto = np.asarray(g, float), P, sigma2, presupuesto
        self.n_eval = 0
        self.historial = []

    def __call__(self, x):
        X = np.atleast_2d(x)
        if self.n_eval + X.shape[0] > self.presupuesto:
            raise RuntimeError("Se superó el presupuesto de evaluaciones")
        f = -capacidad(reparar(X, self.P), self.g, self.sigma2)
        self.n_eval += X.shape[0]
        self.historial.extend(f.tolist())
        return float(f[0]) if np.ndim(x) == 1 else f


def water_filling(g=G, P=P_TOTAL, sigma2=SIGMA2, tol=1e-14):
    """Solución exacta p_i* = max(0, mu - sigma2/g_i); mu por bisección tal que sum p* = P.
    Devuelve (p*, mu, C*)."""
    g = np.asarray(g, float)
    ruido = sigma2 / g
    lo, hi = ruido.min(), ruido.max() + P
    for _ in range(300):
        mu = 0.5 * (lo + hi)
        if np.maximum(0.0, mu - ruido).sum() > P:
            hi = mu
        else:
            lo = mu
        if hi - lo < tol:
            break
    mu = 0.5 * (lo + hi)
    p = np.maximum(0.0, mu - ruido)
    p = P * p / p.sum()                       # ajuste de redondeo: sum p = P exacto
    return p, mu, float(capacidad(p, g, sigma2)[0])


def asignacion_uniforme(g=G, P=P_TOTAL, sigma2=SIGMA2):
    """p_i = P / D y su capacidad."""
    p = np.full(len(g), P / len(g))
    return p, float(capacidad(p, np.asarray(g, float), sigma2)[0])
