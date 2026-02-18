# Codigo (Version Canonica)

Este modulo contiene la implementacion principal de calculo y validacion.

## Archivos canonicos
- `Codigo/Fuente/CsigmaInerteHeegner.cpp`: generador principal para los 9 discriminantes de Heegner.
- `Codigo/Fuente/PipelineCsigma.py`: analisis estadistico y auditoria de CSV.
- `Codigo/Pruebas/ValidarHeegner.py`: validacion matematica local (N, H, identidad de C).
- `Codigo/ReconstruirSalidas.ps1`: pipeline reproducible de compilacion + generacion + validacion.

## Salida canonica
- `Datos/Canonicos/`: CSV finales por discriminante y `ResumenHeegner.csv`.

## Compilar
```powershell
g++ -O3 -march=native -fopenmp -std=c++17 -pipe Codigo/Fuente/CsigmaInerteHeegner.cpp -o Codigo/Binarios/CsigmaHeegner.exe
```

## Generar datos (canonico)
```powershell
./Codigo/Binarios/CsigmaHeegner.exe --all --Rmax 10000001 --step 10 --Rmin 101 --outdir Datos/Canonicos --summary ResumenHeegner.csv
```

## Validar coherencia matematica
```powershell
python Codigo/Pruebas/ValidarHeegner.py --csv-dir Datos/Canonicos --max-R-check 401
```

## Analizar un CSV
```powershell
python Codigo/Fuente/PipelineCsigma.py run Datos/Canonicos/csigma_inerte_D-43_Rmax10000001_step10_Rmin101.csv --outdir Analisis/Csigma --audit_H --audit_N
```

## Convencion operativa
- `C_sigma_inerte(R) = N_R * log(R) / (H_R * R)`.
- `H_R` usa solo factores inertes (incluye `p=2` solo cuando 2 es inerte y divide `R`).
- En corridas canonicas del paper, `R` es impar y la malla es de paso par en `R` (ejemplo: `deltaR=20` cuando `step=10`).
