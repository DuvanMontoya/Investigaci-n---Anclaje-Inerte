param(
    [int]$Rmax = 10000001,
    [int]$Step = 10,
    [int]$Rmin = 101,
    [int]$MaxRCheck = 401
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$codigoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $codigoDir
$exeRel = "Codigo/Binarios/CsigmaHeegner.exe"
$srcRel = "Codigo/Fuente/CsigmaInerteHeegner.cpp"
$outRel = "Datos/Canonicos"
$validateRel = "Codigo/Pruebas/ValidarHeegner.py"
$auditRel = "Herramientas/AuditarProyecto.py"

Push-Location $rootDir
try {
    $exeDir = Split-Path -Parent $exeRel
    if (-not (Test-Path -LiteralPath $exeDir)) {
        New-Item -ItemType Directory -Path $exeDir | Out-Null
    }

    Write-Host "==> Compilando binario canonico..."
    g++ -O3 -march=native -fopenmp -std=c++17 -pipe $srcRel -o $exeRel
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo compilacion (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Limpiando salida previa..."
    if (Test-Path $outRel) {
        Remove-Item -Recurse -Force $outRel
    }
    New-Item -ItemType Directory -Path $outRel | Out-Null

    Write-Host "==> Regenerando Datos/Canonicos desde cero..."
    & ".\$exeRel" --all --Rmax $Rmax --step $Step --Rmin $Rmin --outdir $outRel --summary "ResumenHeegner.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo generacion C++ (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Validacion matematica profunda sobre Datos/Canonicos..."
    python $validateRel --csv-dir $outRel --max-R-check $MaxRCheck
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo ValidarHeegner.py (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Auditoria integral del proyecto..."
    python $auditRel
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo AuditarProyecto.py (exit=$LASTEXITCODE)."
    }

    Write-Host "==> Proceso completado OK."
    Write-Host "Comando reproducible:"
    Write-Host "powershell -ExecutionPolicy Bypass -File Codigo/ReconstruirSalidas.ps1"
}
finally {
    Pop-Location
}
