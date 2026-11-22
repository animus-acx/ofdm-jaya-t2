"""Pruebas de funcionalidad del trabajo T2 (modelo, water-filling, JAYA, protocolo) y traza
manual de una generación de JAYA en un caso pequeño (2 subportadoras, 3 individuos).

Uso:  python pruebas.py       (debe terminar con "TODAS LAS PRUEBAS PASARON")
"""
import math
import sys

import numpy as np

from ofdm import (G, P_TOTAL, SIGMA2, D, reparar, capacidad, Objetivo, water_filling, asignacion_uniforme)
from jaya import jaya, actualizar_jaya
from busqueda_aleatoria import busqueda_aleatoria
from protocolo import correr

RESULTADOS = []


def ok(condicion, mensaje):
    RESULTADOS.append(bool(condicion))
    print(("[OK]     " if condicion else "[FALLA]  ") + mensaje)


def f_escalar(x, g, P=1.0, s2=0.1):
    """Función objetivo calculada con math y ciclos for (verificación independiente, a mano)."""
    s = sum(x)
    p = [P / len(x)] * len(x) if s == 0 else [P * xi / s for xi in x]
    return -sum(math.log2(1 + gi * pi / s2) for gi, pi in zip(g, p))


# ---------------------------------------------------------------- 1. Modelo
def pruebas_modelo():
    print("\n--- 1. Modelo y función objetivo ---")
    rng = np.random.default_rng(0)
    X = rng.random((1000, D))
    p = reparar(X)
    ok(np.allclose(p.sum(axis=1), P_TOTAL, atol=1e-12) and (p >= 0).all(), "la reparación da sum(p) = P y p >= 0 en 1000 puntos aleatorios")
    ok(np.allclose(reparar(np.zeros(D)), P_TOTAL / D), "x = 0 se repara a potencia uniforme")
    x = rng.random(D)
    obj = Objetivo()
    ok(abs(obj(x) - obj(7.3 * x)) < 1e-12, "f(λx) = f(x): la función es invariante a la escala de x")
    # evaluación a mano: asignación uniforme
    c_unif = sum(math.log2(1 + 0.625 * gi) for gi in G)
    ok(abs(c_unif - 9.565150) < 5e-7, f"C uniforme a mano = {c_unif:.6f} (esperado 9.565150)")
    ok(abs(Objetivo()(np.ones(D)) + c_unif) < 1e-12, "f(x = 1,...,1) del código coincide con el cálculo a mano")
    # toda la potencia en la subportadora 12
    e12 = np.zeros(D); e12[11] = 1.0
    ok(abs(Objetivo()(e12) + math.log2(1 + 2.782 / 0.1)) < 1e-12, f"f(e_12) = -log2(1 + 2.782/0.1) = {-math.log2(1 + 2.782/0.1):.6f}")
    # contador y presupuesto
    obj = Objetivo(presupuesto=60)
    obj(rng.random((30, D))); obj(rng.random((30, D)))
    try:
        obj(rng.random(D)); lanzo = False
    except RuntimeError:
        lanzo = True
    ok(obj.n_eval == 60 and lanzo and len(obj.historial) == 60, "el contador cuenta 60 evaluaciones y bloquea la 61")
    ok(isinstance(Objetivo()(rng.random(D)), float), "un vector de entrada devuelve un escalar")
    # función con ciclos for == vectorizada
    X = rng.random((20, D))
    dif = max(abs(f_escalar(list(xi), list(G)) - fi) for xi, fi in zip(X, Objetivo()(X)))
    ok(dif < 1e-12, f"función vectorizada == versión con ciclos for (dif. máx. {dif:.1e})")


# ---------------------------------------------------------------- 2. Water-filling
def pruebas_water_filling():
    print("\n--- 2. Water-filling (referencia exacta) ---")
    p, mu, c_star = water_filling()
    ruido = SIGMA2 / G
    ok(abs(p.sum() - P_TOTAL) < 1e-12 and (p >= 0).all(), "sum(p*) = P y p* >= 0")
    ok(np.where(p == 0)[0].tolist() == [0, 2, 7, 13, 14], "subportadoras apagadas: 1, 3, 8, 14 y 15")
    act = p > 0
    ok(np.allclose(p[act] + ruido[act], mu, atol=1e-12) and (ruido[~act] >= mu).all(), "KKT: p_i + sigma²/g_i = mu en las activas y sigma²/g_i >= mu en las apagadas")
    # método a mano: conjunto activo = las k subportadoras de menor sigma²/g; mu_k = (P + sum sigma²/g)/k
    orden = np.argsort(ruido)
    valido = None
    for k in range(1, D + 1):
        mu_k = (P_TOTAL + ruido[orden[:k]].sum()) / k
        sig = ruido[orden[k]] if k < D else np.inf
        if ruido[orden[k - 1]] < mu_k <= sig:
            valido = (k, mu_k)
    ok(valido is not None and valido[0] == 11 and abs(valido[1] - mu) < 1e-12, f"método del conjunto activo (a mano): k = {valido[0]}, mu = {valido[1]:.6f}")
    c_mano = sum(math.log2(mu * G[i] / SIGMA2) for i in np.where(act)[0])
    ok(abs(c_mano - c_star) < 1e-9, f"C* = sum log2(mu·g_i/sigma²) sobre las activas = {c_mano:.6f}")
    # optimalidad: ninguna asignación factible supera a C*
    rng = np.random.default_rng(1)
    pr = rng.dirichlet(np.ones(D), 100000) * P_TOTAL
    pr2 = rng.dirichlet(0.2 * np.ones(D), 100000) * P_TOTAL            # asignaciones dispersas
    ok(capacidad(pr).max() < c_star and capacidad(pr2).max() < c_star, "100 000 + 100 000 asignaciones aleatorias factibles no superan a C*")
    # óptimo local: perturbaciones factibles pequeñas de p* reducen C
    mejoras = 0
    for _ in range(2000):
        d = rng.normal(size=D); d -= d.mean(); q = np.maximum(p + 1e-3 * d, 0); q *= P_TOTAL / q.sum()
        mejoras += capacidad(q)[0] > c_star + 1e-12
    ok(mejoras == 0, "2 000 perturbaciones factibles de p* no aumentan la capacidad")
    p_u, c_u = asignacion_uniforme()
    ok(c_u < c_star, f"la asignación uniforme ({c_u:.4f}) es peor que el óptimo ({c_star:.4f})")


# ---------------------------------------------------------------- 3. JAYA
def pruebas_jaya():
    print("\n--- 3. JAYA y protocolo ---")
    rng = np.random.default_rng(2)
    X, r1, r2 = rng.random((30, D)), rng.random((30, D)), rng.random((30, D))
    mejor, peor = X[3], X[11]
    vec = actualizar_jaya(X, mejor, peor, r1, r2)
    ciclo = np.array([[X[i, j] + r1[i, j] * (mejor[j] - abs(X[i, j])) - r2[i, j] * (peor[j] - abs(X[i, j])) for j in range(D)] for i in range(30)])
    ok(np.allclose(vec, ciclo, atol=1e-14), "la ecuación de JAYA vectorizada == versión con ciclos for")
    r = correr(jaya, 1)
    ok(r["n_eval"] == 3000 and len(r["curva"]) == 3000, "una corrida de JAYA usa exactamente 3000 evaluaciones")
    ok((r["x"] >= 0).all() and (r["x"] <= 1).all(), "la solución final está dentro de [0, 1]")
    ok((np.diff(r["curva"]) <= 0).all(), "la curva de convergencia no aumenta")
    ok(abs(Objetivo()(r["x"]) - r["f"]) < 1e-12, "mejor_f coincide con f(mejor_x)")
    r_b = correr(jaya, 1)
    ok(r_b["f"] == r["f"] and np.array_equal(r_b["x"], r["x"]), "misma semilla -> mismo resultado (reproducible)")
    ok(correr(jaya, 2)["f"] != r["f"], "semillas distintas -> resultados distintos")
    for n in (10, 20, 30, 50, 100):
        ok(correr(jaya, 1, n_pob=n, guardar_curva=False)["n_eval"] == 3000, f"N = {n}: {3000 // n} generaciones, 3000 evaluaciones")
    c_star = water_filling()[2]
    brechas = [c_star + correr(jaya, s, guardar_curva=False)["f"] for s in range(1, 6)]
    ok(max(brechas) < 0.05 and min(brechas) >= -1e-9, f"brecha respecto del óptimo en 5 semillas: {max(brechas):.4f} máx. (nunca negativa)")
    f_alea = correr(busqueda_aleatoria, 1, guardar_curva=False)["f"]
    ok(r["f"] < f_alea, f"JAYA ({r['f']:.4f}) mejora a la búsqueda aleatoria ({f_alea:.4f}) con la misma semilla")


# ---------------------------------------------------------------- 4. Traza manual de JAYA (caso pequeño)
G2, S2_ = np.array([1.0, 0.5]), 0.1
X0 = np.array([[0.2, 0.9], [0.6, 0.3], [0.8, 0.5]])
R1 = np.array([[0.3, 0.7], [0.9, 0.2], [0.5, 0.6]])
R2 = np.array([[0.4, 0.1], [0.8, 0.5], [0.2, 0.9]])


def traza_jaya(imprimir=True):
    f = lambda X: np.array([f_escalar(list(x), list(G2), 1.0, S2_) for x in X])
    fit = f(X0)
    ib, iw = int(np.argmin(fit)), int(np.argmax(fit))
    mejor, peor = X0[ib], X0[iw]
    Xs = actualizar_jaya(X0, mejor, peor, R1, R2)
    Xn = np.clip(Xs, 0, 1)
    fn = f(Xn)
    acepta = fn < fit
    Xf = np.where(acepta[:, None], Xn, X0)
    if imprimir:
        print("\n--- 4. Traza manual de una generación de JAYA (2 subportadoras, 3 individuos) ---")
        print("g = (1.0, 0.5), sigma² = 0.1, P = 1;  f = -C;  r1 y r2 explícitos")
        for i in range(3):
            print(f"  ind {i+1}: x = {X0[i]}  p = {X0[i]/X0[i].sum()}  f = {fit[i]:.6f}")
        print(f"  mejor = individuo {ib+1} {mejor},  peor = individuo {iw+1} {peor}")
        for i in range(3):
            for j in range(2):
                print(f"  ind {i+1}, x{j+1}: {X0[i,j]} + {R1[i,j]}·({mejor[j]} - |{X0[i,j]}|) - {R2[i,j]}·({peor[j]} - |{X0[i,j]}|) = {Xs[i,j]:.6f} -> recortado {Xn[i,j]:.6f}")
        for i in range(3):
            print(f"  ind {i+1}: candidato {Xn[i]}  f' = {fn[i]:.6f}  vs f = {fit[i]:.6f}  -> {'ACEPTA' if acepta[i] else 'rechaza'}")
        print("  población nueva:", Xf.tolist())
    return Xs, Xn, fit, fn, Xf


def pruebas_traza():
    Xs, Xn, fit, fn, Xf = traza_jaya()
    # valores calculados a mano con calculadora (4 decimales); reemplácelos por los de su hoja si difieren
    fit_mano = np.array([-3.8427, -4.3536, -4.3862])
    Xn_mano = np.array([[0.38, 0.62], [1.0, 0.04], [0.92, 0.14]])
    fn_mano = np.array([-4.2987, -3.6618, -4.0064])
    Xf_mano = np.array([[0.38, 0.62], [0.6, 0.3], [0.8, 0.5]])
    ok(np.allclose(fit, fit_mano, atol=1e-4), "traza: aptitud inicial código == mano (4 decimales)")
    ok(np.allclose(Xn, Xn_mano, atol=1e-4), "traza: candidatos recortados código == mano")
    ok(np.allclose(fn, fn_mano, atol=1e-4), "traza: aptitud de los candidatos código == mano")
    ok(np.allclose(Xf, Xf_mano, atol=1e-4), "traza: población nueva (selección voraz) código == mano")
    ok(np.allclose(Xs, actualizar_jaya(X0, X0[np.argmin(fit)], X0[np.argmax(fit)], R1, R2)), "traza: candidatos == actualizar_jaya del código")


if __name__ == "__main__":
    pruebas_modelo()
    pruebas_water_filling()
    pruebas_jaya()
    pruebas_traza()
    print("\n" + ("TODAS LAS PRUEBAS PASARON" if all(RESULTADOS) else "HAY PRUEBAS QUE FALLARON"),
          f"({sum(RESULTADOS)} de {len(RESULTADOS)})")
    sys.exit(0 if all(RESULTADOS) else 1)
