param(
    [int]$Rmax = 10000001,
    [int]$Step = 10,
    [int]$Rmin = 101,
    [int]$MaxRCheck = 401
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$rootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Reset-Directory {
    param([Parameter(Mandatory = $true)][string]$PathRel)
    $abs = Join-Path $rootDir $PathRel
    if (-not (Test-Path -LiteralPath $abs)) {
        New-Item -ItemType Directory -Path $abs | Out-Null
        return
    }
    Get-ChildItem -LiteralPath $abs -Force | Remove-Item -Recurse -Force
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$PathRel)
    $abs = Join-Path $rootDir $PathRel
    if (-not (Test-Path -LiteralPath $abs)) {
        New-Item -ItemType Directory -Path $abs | Out-Null
    }
}

Push-Location $rootDir
try {
    Write-Host "==> Limpieza completa de salidas..."
    Reset-Directory "Datos/Canonicos"
    Reset-Directory "Datos/Generados"
    Reset-Directory "Analisis/Csigma"
    Reset-Directory "Figuras/Articulo"
    Reset-Directory "Manuscrito/Salida"
    Reset-Directory "Manuscrito/Temporal"
    Ensure-Directory "Figuras/Articulo/Extra"
    Ensure-Directory "Figuras/Articulo/Diagnosticos"

    Write-Host "==> Compilando C++..."
    Ensure-Directory "Codigo/Binarios"
    g++ -O3 -march=native -fopenmp -std=c++17 -pipe Codigo/Fuente/CsigmaInerteHeegner.cpp -o Codigo/Binarios/CsigmaHeegner.exe
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo compilacion C++ (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Generando CSV canonicos..."
    & ".\Codigo\Binarios\CsigmaHeegner.exe" --all --Rmax $Rmax --step $Step --Rmin $Rmin --outdir "Datos/Canonicos" --summary "ResumenHeegner.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo generacion canonica C++ (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Validando CSV canonicos..."
    python Codigo/Pruebas/ValidarHeegner.py --csv-dir "Datos/Canonicos" --max-R-check $MaxRCheck
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo ValidarHeegner.py (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Generando CSV auxiliares (Python)..."
    python Herramientas/GenerarConteosPrimarios.py --output_dir "Datos/Generados" --grid both --R_min 101 --R_max 401 --method hybrid --seed 20260217
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo GenerarConteosPrimarios.py (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Generando figuras del articulo..."
    python Herramientas/GenerarFigurasArticulo.py --input_dir "Datos/Canonicos" --bundle_dir "Figuras/Articulo"
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo GenerarFigurasArticulo.py (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Generando analisis por discriminante..."
    $csvs = Get-ChildItem -LiteralPath "Datos/Canonicos" -File -Filter "csigma_inerte_D-*_Rmax*_step*_Rmin*.csv" | Sort-Object Name
    foreach ($csv in $csvs) {
        $m = [regex]::Match($csv.Name, "D-?\d+")
        $tag = if ($m.Success) { $m.Value } else { [System.IO.Path]::GetFileNameWithoutExtension($csv.Name) }
        $outdir = Join-Path "Analisis/Csigma" $tag
        Ensure-Directory $outdir

        $args = @(
            "-X", "utf8",
            "Codigo/Fuente/PipelineCsigma.py", "run", $csv.FullName,
            "--outdir", $outdir,
            "--blocks", "200",
            "--minR", "1000000",
            "--weights", "n",
            "--audit_H"
        )
        if ($tag -eq "D-4") { $args += "--audit_N" }
        python @args
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo PipelineCsigma.py en $($csv.Name) (exit=$LASTEXITCODE)."
        }
    }

    Write-Host "==> Compilando manuscrito y dejando solo PDF..."
    powershell -ExecutionPolicy Bypass -File "Manuscrito/CompilarManuscrito.ps1" -LimpiarAuxiliares
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo CompilarManuscrito.ps1 (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Auditoria final..."
    python Herramientas/AuditarProyecto.py
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo AuditarProyecto.py (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Regeneracion completa finalizada correctamente."
    Write-Host "Comando para repetir:"
    Write-Host "powershell -ExecutionPolicy Bypass -File Herramientas/RegenerarTodo.ps1"
}
finally {
    Pop-Location
}
