param(
    [switch]$LimpiarAuxiliares
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$manuscritoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fuenteDir = Join-Path $manuscritoDir "Fuente"
$salidaDir = Join-Path $manuscritoDir "Salida"
$temporalDir = Join-Path $manuscritoDir "Temporal"
$texName = "InvestigacionCompleta.tex"
$pdfName = "InvestigacionCompleta.pdf"

if (-not (Test-Path -LiteralPath $salidaDir)) {
    New-Item -ItemType Directory -Path $salidaDir | Out-Null
}
if (-not (Test-Path -LiteralPath $temporalDir)) {
    New-Item -ItemType Directory -Path $temporalDir | Out-Null
}

Push-Location $fuenteDir
try {
    Write-Host "==> Compilando manuscrito en carpeta temporal..."
    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir="../Temporal" $texName
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo compilacion LaTeX (exit=$LASTEXITCODE)."
    }

    $pdfTemporal = Join-Path $temporalDir $pdfName
    $pdfFinal = Join-Path $salidaDir $pdfName
    Move-Item -LiteralPath $pdfTemporal -Destination $pdfFinal -Force
    Write-Host "==> PDF actualizado en: $pdfFinal"

    if ($LimpiarAuxiliares) {
        Write-Host "==> Limpiando auxiliares de LaTeX..."
        latexmk -c -outdir="../Temporal" $texName | Out-Null
    }
}
finally {
    Pop-Location
}
