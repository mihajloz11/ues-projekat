$ErrorActionPreference = "Stop"

$espPatterns = "VID_303A|VID_10C4|VID_1A86|VID_0403|CP210|CH340|CH910|UART|JTAG|CDC|Espressif|ESP32|USB Serial|USB-SERIAL"

Write-Host "Serijski portovi:"
$ports = Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -match "\(COM[0-9]+\)" -or $_.PNPDeviceID -match $espPatterns -or $_.Name -match $espPatterns } |
    Select-Object Status, ConfigManagerErrorCode, Name, PNPDeviceID

if ($ports) {
    $ports | Format-Table -AutoSize
} else {
    Write-Host "Nijedan ESP ili serijski COM port nije pronađen."
    Write-Host ""
    Write-Host "Mogući uzroci: USB data kabl, USB port, drajver ili ESP32-S3 BOOT/RESET mod."
}
