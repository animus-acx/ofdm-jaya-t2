"""JAYA - Rao (2016), Int. J. Industrial Engineering Computations 7(1):19-34.

  X'_ij = X_ij + r1_j * (X_best,j - |X_ij|) - r2_j * (X_worst,j - |X_ij|)       (ec. de JAYA)
  r1_j, r2_j ~ U(0, 1).  Selección voraz: X_i <- X'_i solo si f(X'_i) < f(X_i).
JAYA no tiene parámetros propios (solo el tamaño de población y el número de generaciones).
El mejor y el peor se determinan al inicio de cada generación (actualización síncrona).
"""
import numpy as np


def actualizar_jaya(X, mejor, peor, r1, r2):
    """Candidatos de toda la población (vectorizado, sin recortar)."""
    return X + r1 * (mejor - np.abs(X)) - r2 * (peor - np.abs(X))


def jaya(f, lim_inf, lim_sup, dim, n_pob, n_gen, semilla):
    """JAYA con la interfaz común. n_gen = generaciones TOTALES, contando la inicial,
    de modo que las evaluaciones son n_pob * n_gen (= 3000 con n_gen = 3000 / n_pob).
    Devuelve mejor_x, mejor_f, curva (mejor f al final de cada generación)."""
    rng = np.random.default_rng(semilla)
    X = rng.uniform(lim_inf, lim_sup, (n_pob, dim))          # generación 0
    fit = f(X)
    curva = np.empty(n_gen)
    curva[0] = fit.min()

    for g in range(1, n_gen):
        mejor = X[np.argmin(fit)]
        peor = X[np.argmax(fit)]
        r1 = rng.random((n_pob, dim))
        r2 = rng.random((n_pob, dim))
        Xn = np.clip(actualizar_jaya(X, mejor, peor, r1, r2), lim_inf, lim_sup)
        fn = f(Xn)
        mejora = fn < fit                                    # selección voraz
        X[mejora], fit[mejora] = Xn[mejora], fn[mejora]
        curva[g] = fit.min()
    ib = np.argmin(fit)
    return X[ib].copy(), float(fit[ib]), curva
