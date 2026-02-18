# Anclaje Inerte - Investigacion 2

Estructura profesional y unificada del proyecto.

## Estructura
- `Codigo/`
  - `Fuente/`: implementacion principal (`CsigmaInerteHeegner.cpp`, `PipelineCsigma.py`).
  - `Pruebas/`: validaciones (`ValidarHeegner.py`) y bancos de prueba.
  - `Binarios/`: ejecutables locales (ignorado por Git).
  - `ReconstruirSalidas.ps1`: flujo reproducible de compilacion + generacion + auditoria.
- `Datos/`
  - `Canonicos/`: unica fuente canonica de CSV finales.
  - `Generados/`: corridas experimentales/controladas.
  - `Auxiliares/`: utilidades secundarias de verificacion.
- `Analisis/`
  - `Csigma/`: salidas del pipeline estadistico.
- `Figuras/`
  - `Articulo/`: figuras y tablas usadas por el manuscrito.
- `Manuscrito/`
  - `Fuente/`: `InvestigacionCompleta.tex`.
  - `Salida/`: PDF final.
  - `Temporal/`: auxiliares de compilacion LaTeX (ignorado por Git).
- `Temporal/`: scratch, pruebas y material legado (ignorado por Git).

## Secuencia recomendada
1. Compilar y generar datos canonicos:
```powershell
powershell -ExecutionPolicy Bypass -File Codigo/ReconstruirSalidas.ps1
```

2. Generar figuras del articulo:
```powershell
python Herramientas/GenerarFigurasArticulo.py --input_dir Datos/Canonicos --bundle_dir Figuras/Articulo
```

3. Auditar consistencia global:
```powershell
python Herramientas/AuditarProyecto.py
```

4. Compilar manuscrito sin ensuciar raiz:
```powershell
powershell -ExecutionPolicy Bypass -File Manuscrito/CompilarManuscrito.ps1 -LimpiarAuxiliares
```
# Investigaci-n---Anclaje-Inerte
