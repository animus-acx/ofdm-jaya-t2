# T2 – Asignación de potencia en subportadoras OFDM con JAYA

Problema: 16 subportadoras, P = 1 W, sigma² = 0.1 W; se maximiza la capacidad
C(p) = sum log2(1 + g_i p_i / sigma²) con el operador de reparación p_i = P x_i / sum x_j
y x en [0, 1]^16 (f = -C). Algoritmo asignado: JAYA (Rao, 2016). Presupuesto: 3 000 evaluaciones por corrida.

## Archivos
- `ofdm.py`  Modelo: datos g_i, reparación, capacidad, función objetivo con contador de evaluaciones, water-filling exacto (bisección) y asignación uniforme.
- `jaya.py`  JAYA implementado a partir del artículo original (vectorizado).
- `busqueda_aleatoria.py`  Línea base: búsqueda aleatoria pura con el mismo presupuesto.
- `protocolo.py`  Corrida con contador, estadística, tablas y canales Rayleigh.
- `validacion.py`  Validación V1 (water-filling), V2 (KKT) y V3 (SNR, Rayleigh, dimensión).
- `pruebas.py`  Pruebas de funcionalidad (modelo, water-filling, JAYA, protocolo) y traza manual de una generación de JAYA.
- `main.py`  Ejecuta todo el protocolo y la validación y guarda figuras (PNG), tablas (CSV) y `resultados.json` en `resultados/`.

## Cómo reproducir todos los resultados
    pip install numpy scipy matplotlib
    python pruebas.py   # verifica el funcionamiento (debe terminar con TODAS LAS PRUEBAS PASARON)
    python main.py

Semillas: 1 a 30 para JAYA, búsqueda aleatoria y sensibilidad; los canales Rayleigh usan las
semillas 1000·D + r (r = 1..20). SciPy se usa solo para la prueba de Wilcoxon (`scipy.stats.wilcoxon`).
En Spyder: abrir `main.py` y pulsar F5 (con `MOSTRAR = True` se ven las figuras en pantalla).
