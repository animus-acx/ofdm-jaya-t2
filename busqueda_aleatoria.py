"""Línea base: búsqueda aleatoria pura (puntos uniformes en la caja), mismo presupuesto."""
import numpy as np


def busqueda_aleatoria(f, lim_inf, lim_sup, dim, n_pob, n_gen, semilla):
    """n_gen bloques de n_pob puntos uniformes (n_pob * n_gen evaluaciones = 3000).
    Misma interfaz que jaya: mejor_x, mejor_f, curva (mejor f al final de cada bloque)."""
    rng = np.random.default_rng(semilla)
    mejor_x, mejor_f = None, np.inf
    curva = np.empty(n_gen)
    for g in range(n_gen):
        X = rng.uniform(lim_inf, lim_sup, (n_pob, dim))
        fit = f(X)
        ib = np.argmin(fit)
        if fit[ib] < mejor_f:
            mejor_x, mejor_f = X[ib].copy(), float(fit[ib])
        curva[g] = mejor_f
    return mejor_x, mejor_f, curva
