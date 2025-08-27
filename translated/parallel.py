import argparse, os, sys, time, math, random, re, platform, subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run(cmd, **kwargs):
    check = kwargs.pop("check", True)
    env = kwargs.pop("env", None)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, **kwargs)
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
    return p.returncode, p.stdout, p.stderr

def detect_sources():
    base = Path.cwd()
    cand = base / "original"
    if (cand / "serial.c").exists() and (cand / "paralelo.c").exists():
        return cand / "serial.c", cand / "paralelo.c", (base / "bin")
    if (base / "serial.c").exists() and (base / "paralelo.c").exists():
        return base / "serial.c", base / "paralelo.c", (base / "bin")
    raise FileNotFoundError("No encontré serial.c y paralelo.c ni en ./original/ ni en la raíz del proyecto.")

def openmp_flags():
    sysname = platform.system()
    if sysname == "Darwin":
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    else:
        return ["-fopenmp"], []

def parse_time_threads(text):
    t = None
    m = re.search(r"Tiempo total.*?=\s*([0-9]+\.[0-9]+)\s*s", text)
    if m:
        t = float(m.group(1))
    p = None
    m2 = re.search(r"\((?:paralelo|python-threads),\s*([0-9]+)\s*hilos\)", text)
    if m2:
        p = int(m2.group(1))
    m3 = re.search(r"\(paralelo,\s*([0-9]+)\s*hilos\)", text)
    if m3:
        p = int(m3.group(1))
    return t, p

def generate_input(path, R, T, seed=42):
    random.seed(seed)
    path = Path(path)
    with path.open("w") as f:
        for r in range(R):
            base_temp = random.uniform(18, 30)
            base_hum  = random.uniform(40, 80)
            base_wind = random.uniform(2, 12)

            trend_temp = random.uniform(-0.01, 0.02)
            trend_hum  = random.uniform(-0.03, 0.03)
            trend_wind = random.uniform(-0.01, 0.01)

            for t in range(T):
                temp = base_temp + trend_temp * t + random.gauss(0, 0.3)
                hum  = max(0.0, min(100.0, base_hum + trend_hum * t + random.gauss(0, 1.2)))
                wind = max(0.0, base_wind + trend_wind * t + random.gauss(0, 0.2))
                f.write(f"{r} {temp:.6f} {hum:.6f} {wind:.6f}\n")
    return path

def slope_last_k(a, K):
    if K < 2:
        return 0.0
    if K > len(a):
        K = len(a)
    start_idx = len(a) - K
    sumx = K * (K - 1) // 2
    sumx2 = (K - 1) * K * (2*K - 1) // 6
    sumy = 0.0
    sumxy = 0.0
    for i in range(K):
        y = a[start_idx + i]
        sumy += y
        sumxy += i * y
    denom = K * sumx2 - sumx * sumx
    if abs(denom) < 1e-12:
        return 0.0
    return (K * sumxy - sumx * sumy) / denom

def py_parallel_threads(input_path, F, threads):
    regions = []
    temps, hums, winds = [], [], []
    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            r = int(parts[0]); t = float(parts[1]); h = float(parts[2]); v = float(parts[3])
            regions.append(r); temps.append(t); hums.append(h); winds.append(v)
    if not regions:
        raise ValueError("Archivo de input vacío")

    R = max(regions) + 1
    n = len(regions)
    T = n // R
    K = T

    idxs_per_r = [[] for _ in range(R)]
    for idx, r in enumerate(regions):
        idxs_per_r[r].append(idx)

    from collections import deque
    temp = [deque(maxlen=T) for _ in range(R)]
    hum  = [deque(maxlen=T) for _ in range(R)]
    wind = [deque(maxlen=T) for _ in range(R)]
    for r in range(R):
        for pos in idxs_per_r[r]:
            temp[r].append(temps[pos])
            hum[r].append(hums[pos])
            wind[r].append(winds[pos])

    def step_region(r):
        mT = slope_last_k(temp[r], K)
        mH = slope_last_k(hum[r],  K)
        mV = slope_last_k(wind[r], K)
        t_pred = temp[r][-1] + mT
        h_pred = hum[r][-1] + mH
        v_pred = wind[r][-1] + mV
        if h_pred < 0: h_pred = 0.0
        if h_pred > 100: h_pred = 100.0
        if len(temp[r]) == T: temp[r].popleft()
        if len(hum[r])  == T: hum[r].popleft()
        if len(wind[r]) == T: wind[r].popleft()
        temp[r].append(t_pred); hum[r].append(h_pred); wind[r].append(v_pred)

    t0 = time.perf_counter()
    for _ in range(F):
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(step_region, range(R)))
    elapsed = time.perf_counter() - t0
    print(f"Tiempo total (python-threads,{threads} hilos) = {elapsed:.6f} s")
    return elapsed, threads


def write_results(T_serial, T_c, p_c, T_py, p_py):
    S_c = T_serial / T_c
    E_c = S_c / (p_c if p_c and p_c > 0 else 1)

    factor_c_vs_py = T_py / T_c     
    speedup_py_vs_c = T_c / T_py     
    E_py_rel = speedup_py_vs_c / (p_py if p_py and p_py > 0 else 1)

    with open("RESULTS.csv","w") as f:
        f.write("Metric,Value\n")
        f.write(f"T_serial (s),{T_serial:.6f}\n")
        f.write(f"T_parallel_C (s),{T_c:.6f}\n")
        f.write(f"Threads_C,{p_c if p_c else 'N/A'}\n")
        f.write(f"Speedup_C = T_serial/T_parallel_C,{S_c:.3f}\n")
        f.write(f"Eficiencia_C = Speedup_C/Threads_C,{E_c:.3f}\n")
        f.write(f"T_parallel_Python (s),{T_py:.6f}\n")
        f.write(f"Threads_Python,{p_py if p_py else 'N/A'}\n")
        f.write(f"C_vs_Python_Factor = T_py/T_c,{factor_c_vs_py:.3f}\n")
        f.write(f"Python_vs_C_Speedup = T_c/T_py,{speedup_py_vs_c:.3f}\n")
        f.write(f"Python_Relative_Efficiency = (T_c/T_py)/Threads_Python,{E_py_rel:.3f}\n")

    md = f"""# Métricas de Speedup y Eficiencia

**Resultados brutos**
- T_serial = {T_serial:.6f} s
- T_parallel_C (OpenMP, p={p_c}) = {T_c:.6f} s
- T_parallel_Python (threads={p_py}) = {T_py:.6f} s

**C paralelo vs serial**
- Speedup_C = T_serial / T_parallel_C = **{S_c:.3f}**
- Eficiencia_C = Speedup_C / p = **{E_c:.3f}**

**Comparación C paralelo vs Python con hilos**
- Factor (C es X× más rápido que Python) = T_py / T_c = **{factor_c_vs_py:.3f}×**
- Speedup_Python_vs_C (usando C como baseline) = T_c / T_py = **{speedup_py_vs_c:.3f}**  *(<1 implica que Python es más lento)*
- Eficiencia relativa de Python = (T_c/T_py) / threads_python = **{E_py_rel:.3f}**

> Interpretación:
> - Si `Eficiencia_C` se acerca a 1.0, tu paralelización en C está escalando muy bien.
> - Un factor C_vs_Python alto confirma la ventaja de C/OpenMP sobre Python con hilos (GIL).
"""
    with open("metrics_summary.md","w") as f:
        f.write(md)
    print(md)
    print("Guardado: metrics_summary.md y RESULTS.csv")


def build_and_run_c(R, T, F, input_path, c_threads=None):
    serial_c, paralelo_c, bindir = detect_sources()
    bindir.mkdir(parents=True, exist_ok=True)
    serial_bin = bindir / "serial"
    paralelo_bin = bindir / "paralelo"

    CFLAGS = ["-O3", "-march=native", "-Wall"]
    omp_cc, omp_libs = openmp_flags()

    cmd_serial = ["gcc", *CFLAGS, "-o", str(serial_bin), str(serial_c), "-lm"]
    cmd_par    = ["gcc", *CFLAGS, *omp_cc, "-o", str(paralelo_bin), str(paralelo_c), "-lm", *omp_libs]

    print("Compilando serial:", " ".join(cmd_serial))
    run(cmd_serial)
    print("Compilando paralelo:", " ".join(cmd_par))
    try:
        run(cmd_par)
    except subprocess.CalledProcessError as e:
        if platform.system() == "Darwin":
            print("\n[Error] Falló la compilación con OpenMP en macOS.")
            print("Instala libomp:  brew install libomp")
        print(e.stderr)
        raise

    with open(input_path, "r") as fin:
        t0 = time.perf_counter()
        rc, out_s, err_s = run([str(serial_bin), str(F)], input=fin.read())
        T_serial, _ = parse_time_threads(out_s)
        if T_serial is None:
            T_serial = time.perf_counter() - t0
        open("serial.log","w").write(out_s)

    env = os.environ.copy()
    if c_threads:
        env["OMP_NUM_THREADS"] = str(c_threads)
    with open(input_path, "r") as fin:
        t1 = time.perf_counter()
        rc, out_p, err_p = run([str(paralelo_bin), str(F)], input=fin.read(), env=env)
        T_par, p = parse_time_threads(out_p)
        if T_par is None:
            T_par = time.perf_counter() - t1
        open("paralelo.log","w").write(out_p)

    return T_serial, T_par, (p or c_threads or os.cpu_count() or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=32, help="Regiones")
    ap.add_argument("--T", type=int, default=4000, help="Muestras por región")
    ap.add_argument("--F", type=int, default=10, help="Pasos de pronóstico")
    ap.add_argument("--input", type=str, default="datos.txt", help="Archivo de input")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="Hilos para Python")
    ap.add_argument("--c-threads", type=int, default=None, help="Fijar OMP_NUM_THREADS para el C paralelo")
    ap.add_argument("--seed", type=int, default=123, help="Semilla RNG para el input")
    args = ap.parse_args()

    print(f"== Generando input (R={args.R}, T={args.T}) ==")
    generate_input(args.input, args.R, args.T, seed=args.seed)

    print("== Compilando y ejecutando C ==")
    T_serial, T_c, p_c = build_and_run_c(args.R, args.T, args.F, args.input, c_threads=args.c_threads)
    print(f"T_serial: {T_serial:.6f} s")
    print(f"T_parallel_C (p={p_c}): {T_c:.6f} s")

    print("== Ejecutando Python (threads) ==")
    T_py, p_py = py_parallel_threads(args.input, args.F, args.threads)

    print("== Métricas ==")
    write_results(T_serial, T_c, p_c, T_py, p_py)

if __name__ == "__main__":
    main()
