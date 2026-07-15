$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root "tools\.espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe"

if (!(Test-Path $python)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (!$pythonCommand) {
        throw "Python nije pronađen; ESP-IDF okruženje mora biti aktivno."
    }
    $python = $pythonCommand.Source
}

$ports = Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -match "\(COM[0-9]+\)" } |
    ForEach-Object {
        if ($_.Name -match "(COM[0-9]+)") { $Matches[1] }
    } |
    Sort-Object -Unique

if (!$ports) {
    Write-Host "Nijedan COM port nije pronađen."
    exit 1
}

foreach ($port in $ports) {
    Write-Host ""
    Write-Host "=== Bezbjedna provjera porta $port bez flešovanja ==="
    & $python -m esptool --port $port chip_id
}
