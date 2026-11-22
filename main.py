"""Trabajo aplicado T2: asignación de potencia en 16 subportadoras OFDM con JAYA.

Ejecuta TODO el protocolo (30 corridas, semillas 1 a 30, 3 000 evaluaciones por corrida),
la búsqueda aleatoria, la prueba de Wilcoxon, la sensibilidad del tamaño de población N,
la validación V1 a V3 y guarda figuras (PNG), tablas (CSV) y resultados.json en `resultados/`.

Uso:  python main.py          (en Spyder: abrir y pulsar F5)
Requiere: Python >= 3.10, NumPy, SciPy (solo scipy.stats.wilcoxon) y Matplotlib.
"""
import json
import os

import numpy as np
import matplotlib

MOSTRAR = False
if not MOSTRAR:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from ofdm import (G, P_TOTAL, SIGMA2, D, PRESUPUESTO, water_filling, asignacion_uniforme)
from jaya import jaya
from busqueda_aleatoria import busqueda_aleatoria
from protocolo import CARPETA, correr, estadisticas, fmt, guardar_tabla, canal_rayleigh
from validacion import comparar_con_wf, kkt, evaluar_canal, UMBRAL_OFF

SEMILLAS = range(1, 31)                    # semillas 1 a 30
NIVELES_N = [10, 20, 30, 50, 100]          # parámetro clave de JAYA: tamaño de población N
N_BASE = 30                                # 30 x 100 generaciones = 3 000 evaluaciones
SIGMAS = [0.01, 0.1, 1.0]                  # V3a: SNR alta, nominal y baja
N_CANALES = 20                             # V3b: realizaciones Rayleigh
DIMENSIONES = [16, 32, 64]                 # V3c
EV = np.arange(1, PRESUPUESTO + 1)
COL = {"JAYA": "tab:blue", "Búsqueda aleatoria": "tab:gray"}


def guardar_fig(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, nombre), dpi=150)


def fig_convergencia(cj, cr, f_star):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nombre, c in (("JAYA", cj), ("Búsqueda aleatoria", cr)):
        m, s = c.mean(axis=0), c.std(axis=0, ddof=1)
        ax[0].plot(EV, m, color=COL[nombre], lw=1.8, label=f"{nombre} (media)")
        ax[0].fill_between(EV, m - s, m + s, color=COL[nombre], alpha=0.25, label=f"{nombre} ±1 d.e.")
        brecha = c - f_star
        ax[1].plot(EV, np.maximum(brecha.mean(axis=0), 1e-9), color=COL[nombre], lw=1.8, label=f"{nombre} (media)")
        ax[1].fill_between(EV, np.maximum(np.percentile(brecha, 25, axis=0), 1e-9),
                           np.maximum(np.percentile(brecha, 75, axis=0), 1e-9), color=COL[nombre], alpha=0.25,
                           label=f"{nombre} (rango intercuartílico)")
    ax[0].axhline(f_star, color="k", ls="--", lw=1, label=f"f* = -C* = {f_star:.4f}")
    ax[0].set_ylabel("Mejor f = -C (bit/s/Hz)")
    ax[1].set_yscale("log"); ax[1].set_ylabel("Brecha media f - f* = C* - C (bit/s/Hz)")
    for a in ax:
        a.set_xlabel("Evaluaciones de f"); a.grid(alpha=0.3)
    ax[0].set_title("Convergencia: media ±1 desviación estándar (30 corridas)")
    ax[1].set_title("Brecha respecto del óptimo (log.), banda intercuartílica")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    guardar_fig(fig, "fig_convergencia.png")


def fig_sensibilidad(finales, curvas_medias, f_star):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    etiquetas = [str(n) for n in NIVELES_N]
    ax[0].boxplot([finales[n] for n in NIVELES_N]); ax[0].set_xticklabels(etiquetas)
    ax[0].axhline(f_star, color="k", ls="--", lw=1)
    ax[0].set_title("Valor final de f por nivel de N")
    ax[0].set_ylabel("f final = -C (bit/s/Hz)")
    ax[1].boxplot([np.maximum(np.array(finales[n]) - f_star, 1e-9) for n in NIVELES_N]); ax[1].set_xticklabels(etiquetas)
    ax[1].set_yscale("log"); ax[1].set_title("Brecha final C* - C (escala log.)")
    ax[1].set_ylabel("C* - C (bit/s/Hz)")
    for n in NIVELES_N:
        ax[2].plot(EV, np.maximum(curvas_medias[n] - f_star, 1e-9), lw=1.6, label=f"N = {n}")
    ax[2].set_yscale("log"); ax[2].set_xlabel("Evaluaciones de f"); ax[2].set_ylabel("Brecha media C* - C")
    ax[2].set_title("Curvas medias por nivel de N"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].set_xlabel("N"); ax[1].set_xlabel("N")
    guardar_fig(fig, "fig_sensibilidad_N.png")


def fig_v1(p_star, p_jaya, p_unif):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    i = np.arange(1, D + 1); w = 0.27
    ax.bar(i - w, p_star, w, label="Water-filling (exacto)", color="tab:green")
    ax.bar(i, p_jaya, w, label="JAYA (mejor corrida)", color="tab:blue")
    ax.bar(i + w, p_unif, w, label="Uniforme (P/16)", color="tab:gray")
    ax.set_xticks(i); ax.set_xlabel("Subportadora i"); ax.set_ylabel("Potencia p_i (W)")
    ax.set_title("V1: asignación de potencia por subportadora"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    guardar_fig(fig, "fig_v1_potencias.png")


def fig_v2(p_star, mu, p_jaya, k):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    ruido = SIGMA2 / G
    i = np.arange(1, D + 1); tope = 0.9
    for a, p, mu_, titulo in ((ax[0], p_star, mu, "Water-filling exacto"),
                              (ax[1], p_jaya, k["mu_hat"], "JAYA (mejor corrida)")):
        a.bar(i, np.minimum(ruido, tope), color="lightgray", label="sigma²/g_i (fondo)")
        a.bar(i, p, bottom=np.minimum(ruido, tope), color="tab:blue", label="potencia p_i (agua)")
        a.axhline(mu_, color="r", ls="--", lw=1.4, label=f"nivel mu = {mu_:.4f}")
        for j in np.where(ruido > tope)[0]:
            a.text(j + 1, tope - 0.05, f"{ruido[j]:.2f}↑", ha="center", fontsize=7)
        a.set_xticks(i); a.set_xlabel("Subportadora i"); a.set_title(titulo); a.set_ylim(0, tope)
        a.grid(alpha=0.3, axis="y"); a.legend(fontsize=7, loc="upper left")
    ax[0].set_ylabel("Nivel (W)")
    guardar_fig(fig, "fig_v2_waterfilling.png")


def fig_v3a(res):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    i = np.arange(1, D + 1)
    for a, s in zip(ax, SIGMAS):
        r = res[s]
        a.bar(i, r["p_star"], color="tab:green", label="p* (water-filling)")
        a.axhline(P_TOTAL / D, color="k", ls="--", lw=1, label="uniforme P/16")
        a.set_title(f"sigma² = {s} W  ({r['n_off_star']} apagadas)")
        a.set_xlabel("Subportadora i"); a.grid(alpha=0.3, axis="y")
        a.set_xticks(i); a.tick_params(axis="x", labelsize=6)
    ax[0].set_ylabel("Potencia p_i* (W)"); ax[2].legend(fontsize=8, loc="upper left")
    guardar_fig(fig, "fig_v3a_snr.png")


def fig_v3c(tab):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    d = [t["D"] for t in tab]
    ax[0].errorbar(d, [t["gap_media"] for t in tab], yerr=[t["gap_std_canales"] for t in tab], marker="o", capsize=4)
    ax[0].set_xlabel("Número de subportadoras D"); ax[0].set_ylabel("Brecha media C* - C (bit/s/Hz)"); ax[0].set_xticks(d)
    ax[1].errorbar(d, [t["gap_pct_media"] for t in tab], yerr=[t["gap_pct_std_canales"] for t in tab], marker="s", capsize=4, color="tab:red")
    ax[1].set_xlabel("Número de subportadoras D"); ax[1].set_ylabel("Brecha relativa media (%)"); ax[1].set_xticks(d)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].set_title("V3c: brecha absoluta frente a D"); ax[1].set_title("V3c: brecha relativa frente a D")
    guardar_fig(fig, "fig_v3c_dimension.png")


def main():
    p_star, mu, C_star = water_filling()
    f_star = -C_star
    p_unif, C_unif = asignacion_uniforme()
    print("=" * 72)
    print(f"Problema T2: D = {D}, P = {P_TOTAL} W, sigma² = {SIGMA2} W, presupuesto = {PRESUPUESTO}")
    print(f"Water-filling: mu = {mu:.6f}, C* = {C_star:.6f}, apagadas = {(np.where(p_star <= 0)[0] + 1).tolist()}")
    print(f"Asignación uniforme: C = {C_unif:.6f}")
    res = {"C_star": C_star, "mu": mu, "C_unif": C_unif, "p_star": p_star.tolist()}

    # ---------------- Protocolo: JAYA vs búsqueda aleatoria (30 corridas) ----------------
    rj = [correr(jaya, s, n_pob=N_BASE) for s in SEMILLAS]
    rr = [correr(busqueda_aleatoria, s, n_pob=N_BASE) for s in SEMILLAS]
    assert all(r["n_eval"] == PRESUPUESTO for r in rj + rr)
    cj = np.array([r["curva"] for r in rj]); cr = np.array([r["curva"] for r in rr])
    np.save(os.path.join(CARPETA, "curvas_JAYA.npy"), cj)
    np.save(os.path.join(CARPETA, "curvas_aleatoria.npy"), cr)
    fj = np.array([r["f"] for r in rj]); fr = np.array([r["f"] for r in rr])
    est = {"JAYA": estadisticas(fj), "Búsqueda aleatoria": estadisticas(fr)}
    tiempos = {"JAYA": float(np.mean([r["t"] for r in rj])), "Búsqueda aleatoria": float(np.mean([r["t"] for r in rr]))}
    filas = []
    for n, e in est.items():
        filas.append([n, fmt(e["mejor"], 6), fmt(e["mediana"], 6), fmt(e["media"], 6), fmt(e["desv_std"], 6),
                      fmt(e["peor"], 6), fmt(-e["mejor"], 6), fmt(e["media"] - f_star, 6), f"{tiempos[n]*1000:.1f}"])
    guardar_tabla("tabla_estadistica.csv", filas,
                  ["Algoritmo", "Mejor f", "Mediana f", "Media f", "Desv. est. f", "Peor f",
                   "Mejor C (bit/s/Hz)", "Brecha media C*-C", "Tiempo medio por corrida (ms)"])
    w_dos = wilcoxon(fj, fr)
    w_menor = wilcoxon(fj, fr, alternative="less")
    wil = {"estadistico": float(w_dos.statistic), "p_dos_colas": float(w_dos.pvalue),
           "p_JAYA_menor": float(w_menor.pvalue), "n_JAYA_mejor": int(np.sum(fj < fr))}
    guardar_tabla("tabla_wilcoxon.csv", [[fmt(wil["estadistico"], 1), f"{wil['p_dos_colas']:.3e}",
                                          f"{wil['p_JAYA_menor']:.3e}", wil["n_JAYA_mejor"]]],
                  ["Estadístico W", "p (dos colas)", "p (H1: JAYA < aleatoria)", "Semillas con JAYA mejor (de 30)"])
    fig_convergencia(cj, cr, f_star)
    res.update({"estadisticas": est, "tiempos_s": tiempos, "wilcoxon": wil,
                "finales_JAYA": fj.tolist(), "finales_aleatoria": fr.tolist(),
                "media_curva": {"JAYA": {str(k): float(cj.mean(axis=0)[k - 1]) for k in (1, 30, 300, 1000, 2000, 3000)},
                                "aleatoria": {str(k): float(cr.mean(axis=0)[k - 1]) for k in (1, 30, 300, 1000, 2000, 3000)}}})
    print("\n[Protocolo] JAYA:", {k: round(v, 6) for k, v in est["JAYA"].items()})
    print("            aleatoria:", {k: round(v, 6) for k, v in est["Búsqueda aleatoria"].items()})
    print("            Wilcoxon:", wil)

    # ---------------- Sensibilidad del tamaño de población N ----------------
    finales, medias, filas_s, sens = {}, {}, [], {}
    for n in NIVELES_N:
        rs = [correr(jaya, s, n_pob=n) for s in SEMILLAS]
        assert all(r["n_eval"] == PRESUPUESTO for r in rs)
        f = np.array([r["f"] for r in rs]); c = np.array([r["curva"] for r in rs])
        finales[n], medias[n] = f, c.mean(axis=0)
        e = estadisticas(f)
        sens[str(n)] = {**e, "gen": PRESUPUESTO // n, "brecha_media": float(e["media"] - f_star),
                        "n_brecha_menor_1e-3": int(np.sum(f - f_star < 1e-3)),
                        "media_curva": {str(k): float(c.mean(axis=0)[k - 1]) - f_star for k in (100, 300, 1000, 2000, 3000)}}
        filas_s.append([n, PRESUPUESTO // n, fmt(e["mejor"], 6), fmt(e["mediana"], 6), fmt(e["media"], 6),
                        fmt(e["desv_std"], 6), fmt(e["peor"], 6), fmt(e["media"] - f_star, 6),
                        sens[str(n)]["n_brecha_menor_1e-3"]])
    guardar_tabla("tabla_sensibilidad.csv", filas_s,
                  ["N", "Generaciones (3000/N)", "Mejor f", "Mediana f", "Media f", "Desv. est. f", "Peor f",
                   "Brecha media C*-C", "Corridas con brecha < 1e-3"])
    fig_sensibilidad(finales, medias, f_star)
    res["sensibilidad"] = sens
    print("\n[Sensibilidad N]")
    for f_ in filas_s:
        print("  ", f_)

    # ---------------- V1: comparación con water-filling ----------------
    xs = np.array([r["x"] for r in rj])
    v1 = comparar_con_wf(xs)
    ib = int(np.argmin(fj))
    mejor_p = v1["p"][ib]
    fig_v1(p_star, mejor_p, p_unif)
    filas_v1 = [
        ["Water-filling (exacto)", fmt(C_star, 6), "0", "0", "0", f"{int(v1['off_ref'].sum())} apagadas (1, 3, 8, 14, 15)"],
        ["JAYA, mejor corrida (semilla %d)" % (ib + 1), fmt(v1["C"][ib], 6), fmt(v1["gap"][ib], 6),
         fmt(v1["gap_pct"][ib], 6), fmt(v1["dist"][ib], 6), f"{int(v1['n_off_ok'][ib])} de {D} bien clasificadas"],
        ["JAYA, media de 30 corridas", fmt(v1["C"].mean(), 6), fmt(v1["gap"].mean(), 6),
         fmt(v1["gap_pct"].mean(), 6), fmt(v1["dist"].mean(), 6), f"{v1['n_off_ok'].mean():.0f} de {D} bien clasificadas"],
        ["JAYA, peor corrida (semilla %d)" % (int(np.argmax(v1['gap'])) + 1), fmt(v1["C"].min(), 6), fmt(v1["gap"].max(), 6),
         fmt(v1["gap_pct"].max(), 6), fmt(v1["dist"].max(), 6), f"{int(v1['n_off_ok'][int(np.argmax(v1['gap']))])} de {D} bien clasificadas"],
        ["Asignación uniforme", fmt(C_unif, 6), fmt(C_star - C_unif, 6), fmt(100 * (C_star - C_unif) / C_star, 4),
         fmt(np.linalg.norm(p_star - p_unif), 6), "0 apagadas"]]
    guardar_tabla("tabla_v1.csv", filas_v1,
                  ["Asignación", "Capacidad C (bit/s/Hz)", "Brecha ΔC (bit/s/Hz)", "Brecha ΔC (%)",
                   "Distancia ||p - p*|| (W)", "Subportadoras apagadas / clasificación encendida-apagada"])
    res["v1"] = {"gap_mejor": float(v1["gap"][ib]), "gap_pct_mejor": float(v1["gap_pct"][ib]), "dist_mejor": float(v1["dist"][ib]),
                 "gap_media": float(v1["gap"].mean()), "gap_pct_media": float(v1["gap_pct"].mean()),
                 "gap_mediana": float(np.median(v1["gap"])), "gap_max": float(v1["gap"].max()), "gap_min": float(v1["gap"].min()),
                 "dist_media": float(v1["dist"].mean()), "dist_max": float(v1["dist"].max()),
                 "n_off_exacto": int(v1["off_exacto"].sum()), "n_off_ok_media": float(v1["n_off_ok"].mean()),
                 "mejor_semilla": ib + 1, "peor_semilla": int(np.argmax(v1["gap"])) + 1,
                 "dist_unif": float(np.linalg.norm(p_star - p_unif)), "gap_unif": float(C_star - C_unif),
                 "gap_unif_pct": float(100 * (C_star - C_unif) / C_star), "p_mejor": mejor_p.tolist(),
                 "off_ref": (np.where(p_star <= 0)[0] + 1).tolist()}
    print("\n[V1]", {k: v for k, v in res["v1"].items() if k not in ("p_mejor",)})

    # ---------------- V2: condiciones KKT ----------------
    k = kkt(mejor_p)
    kk_all = [kkt(p) for p in v1["p"]]
    fig_v2(p_star, mu, mejor_p, k)
    filas_v2 = [["Water-filling (exacto)", f"{mu:.6f}", "0", "0", f"{int((p_star > 0).sum())}",
                 f"{int((p_star <= 0).sum())} de {int((p_star <= 0).sum())}"],
                ["JAYA, mejor corrida", f"{k['mu_hat']:.6f}", fmt(k["disp_std"], 6), fmt(k["rango"], 6),
                 f"{k['n_activas']}", f"{k['apagadas_ok']} de {k['n_apagadas']}"],
                ["JAYA, media de 30 corridas", f"{np.mean([x['mu_hat'] for x in kk_all]):.6f}",
                 fmt(np.mean([x["disp_std"] for x in kk_all]), 6), fmt(np.mean([x["rango"] for x in kk_all]), 6),
                 f"{np.mean([x['n_activas'] for x in kk_all]):.2f}",
                 f"{np.sum([x['apagadas_ok'] for x in kk_all])} de {np.sum([x['n_apagadas'] for x in kk_all])} (total 30 corridas)"]]
    guardar_tabla("tabla_v2.csv", filas_v2,
                  ["Asignación", "Nivel mu (W)", "Dispersión (d.e. de p_i + sigma²/g_i)", "Rango del nivel (W)",
                   "Subportadoras activas", "Apagadas con sigma²/g_i >= mu"])
    res["v2"] = {"mu_hat_mejor": k["mu_hat"], "disp_std_mejor": k["disp_std"], "rango_mejor": k["rango"],
                 "n_activas_mejor": k["n_activas"], "apagadas_ok_mejor": k["apagadas_ok"], "n_apagadas_mejor": k["n_apagadas"],
                 "disp_std_media": float(np.mean([x["disp_std"] for x in kk_all])),
                 "apagadas_ok_total": int(np.sum([x["apagadas_ok"] for x in kk_all])),
                 "n_apagadas_total": int(np.sum([x["n_apagadas"] for x in kk_all])),
                 "ruido": (SIGMA2 / G).tolist()}
    print("[V2]", {k_: v for k_, v in res["v2"].items() if k_ != "ruido"})

    # ---------------- V3a: SNR alta y baja ----------------
    r3a, filas_a = {}, []
    for s in SIGMAS:
        r = evaluar_canal(G, P_TOTAL, s)
        r3a[s] = r
        filas_a.append([s, fmt(r["C_star"], 4), fmt(r["C_unif"], 4), fmt(r["gan_unif_pct"], 3),
                        fmt(r["dist_unif"], 4), r["n_off_star"], fmt(r["gap"].mean(), 6),
                        fmt(r["gap_pct"].mean(), 6), int(r["off_exacto"].sum())])
    fig_v3a(r3a)
    guardar_tabla("tabla_v3a.csv", filas_a,
                  ["sigma² (W)", "C* (bit/s/Hz)", "C uniforme", "Ganancia de C* sobre uniforme (%)",
                   "||p* - p_unif|| (W)", "Apagadas en el óptimo", "Brecha media JAYA ΔC", "Brecha media JAYA (%)",
                   "Corridas de 30 con apagadas exactas"])
    res["v3a"] = {str(s): {"C_star": r["C_star"], "C_unif": r["C_unif"], "gan_unif_pct": r["gan_unif_pct"],
                           "dist_unif": r["dist_unif"], "n_off_star": r["n_off_star"], "gap_media": float(r["gap"].mean()),
                           "gap_pct_media": float(r["gap_pct"].mean()), "gap_max": float(r["gap"].max()),
                           "n_off_exacto": int(r["off_exacto"].sum())} for s, r in r3a.items()}
    print("[V3a]", filas_a)

    # ---------------- V3b y V3c: Rayleigh y dimensión ----------------
    tab_c, filas_c = [], []
    detalle_b = None
    for dim in DIMENSIONES:
        gaps, gaps_pct, dists, gan, n_off, exactos = [], [], [], [], [], []
        for r_ in range(1, N_CANALES + 1):
            g = canal_rayleigh(dim, 1000 * dim + r_)
            r = evaluar_canal(g)
            gaps.append(r["gap"].mean()); gaps_pct.append(r["gap_pct"].mean()); dists.append(r["dist"].mean())
            gan.append(r["gan_unif_pct"]); n_off.append(r["n_off_star"]); exactos.append(r["off_exacto"].mean())
        t = {"D": dim, "gap_media": float(np.mean(gaps)), "gap_std_canales": float(np.std(gaps, ddof=1)),
             "gap_max_canal": float(np.max(gaps)), "gap_pct_media": float(np.mean(gaps_pct)),
             "gap_pct_std_canales": float(np.std(gaps_pct, ddof=1)), "dist_media": float(np.mean(dists)),
             "gan_unif_pct_media": float(np.mean(gan)), "n_off_media": float(np.mean(n_off)),
             "frac_off_exacto": float(np.mean(exactos))}
        tab_c.append(t)
        filas_c.append([dim, fmt(t["gap_media"], 6), fmt(t["gap_std_canales"], 6), fmt(t["gap_max_canal"], 6),
                        fmt(t["gap_pct_media"], 5), fmt(t["dist_media"], 5), fmt(t["gan_unif_pct_media"], 3),
                        f"{t['n_off_media']:.2f}", f"{100 * t['frac_off_exacto']:.1f}"])
        print(f"[V3 D={dim}]", {k_: round(v, 6) for k_, v in t.items()})
    fig_v3c(tab_c)
    guardar_tabla("tabla_v3bc.csv", filas_c,
                  ["D (subportadoras)", "Brecha media ΔC (bit/s/Hz)", "Desv. est. entre canales", "Peor canal (brecha media)",
                   "Brecha relativa media (%)", "Distancia media ||p - p*||", "Ganancia de C* sobre uniforme (%)",
                   "Apagadas en el óptimo (media)", "Corridas con apagadas exactas (%)"])
    res["v3bc"] = tab_c

    with open(os.path.join(CARPETA, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    if MOSTRAR:
        plt.show()
    print("\nFiguras, tablas y resultados guardados en:", os.path.abspath(CARPETA))


if __name__ == "__main__":
    main()
