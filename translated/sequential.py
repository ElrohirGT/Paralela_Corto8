#!/usr/bin/env python
import sys
import time
import math

def slope_last_k(a, T, K):
    """Calcula la pendiente de regresión lineal usando los últimos K puntos"""
    if K < 2:
        return 0.0
    if K > T:
        K = T
    
    start = T - K
    sumx = sumy = sumxy = sumx2 = 0.0
    
    for i in range(K):
        x = float(i)
        y = a[start + i]
        sumx += x
        sumy += y
        sumxy += x * y
        sumx2 += x * x
    
    denom = K * sumx2 - sumx * sumx
    if abs(denom) < 1e-12:
        return 0.0
    
    return (K * sumxy - sumx * sumy) / denom

def push_next(a, T, next_val):
    """Desplaza el array hacia la izquierda y agrega el nuevo valor al final"""
    for i in range(T - 1):
        a[i] = a[i + 1]
    a[T - 1] = next_val

def main():
    if len(sys.argv) < 2:
        print(f"Uso: {sys.argv[0]} F < datos.txt", file=sys.stderr)
        return 1
    
    F = int(sys.argv[1])
    
    # Leer datos desde stdin
    regions = []
    all_data = []
    max_region = -1
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 4:
                continue
                
            reg = int(parts[0])
            t = float(parts[1])
            h = float(parts[2])
            v = float(parts[3])
            
            regions.append(reg)
            all_data.append([t, h, v])
            
            if reg > max_region:
                max_region = reg
                
    except EOFError:
        pass
    except ValueError as e:
        print(f"Error leyendo datos: {e}", file=sys.stderr)
        return 1
    
    n = len(regions)
    if n == 0:
        print("Archivo vacío", file=sys.stderr)
        return 2
    
    R = max_region + 1  # número de regiones
    T = n // R          # días por región (asumimos balanceado)
    K = T
    
    # Inicializar arrays para cada región
    temp = [[0.0] * T for _ in range(R)]
    hum = [[0.0] * T for _ in range(R)]
    wind = [[0.0] * T for _ in range(R)]
    
    # Contador para llenar los arrays
    count = [0] * R
    
    # Distribuir datos por región
    for i in range(n):
        r = regions[i]
        pos = count[r]
        count[r] += 1
        
        temp[r][pos] = all_data[i][0]
        hum[r][pos] = all_data[i][1]
        wind[r][pos] = all_data[i][2]
    
    # Iniciar medición de tiempo
    t0 = time.time()
    
    # Predicción para F días futuros
    for f in range(F):
        for r in range(R):
            # Calcular pendientes
            mT = slope_last_k(temp[r], T, K)
            mH = slope_last_k(hum[r], T, K)
            mV = slope_last_k(wind[r], T, K)
            
            # Hacer predicciones
            t_pred = temp[r][T-1] + mT
            h_pred = hum[r][T-1] + mH
            v_pred = wind[r][T-1] + mV
            
            # Limitar humedad entre 0 y 100
            if h_pred < 0:
                h_pred = 0
            if h_pred > 100:
                h_pred = 100
            
            # Actualizar arrays con las predicciones
            push_next(temp[r], T, t_pred)
            push_next(hum[r], T, h_pred)
            push_next(wind[r], T, v_pred)
            
            print(f"Region {r} -> Dia+{f+1}: Temp={t_pred:.2f} Hum={h_pred:.2f} Viento={v_pred:.2f}")
    
    # Tiempo final
    t1 = time.time()
    elapsed = t1 - t0
    print(f"Tiempo total (serial) = {elapsed:.6f} s")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
