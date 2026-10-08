param([switch]$Test)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Requirements = Join-Path $ProjectRoot 'requirements.txt'
$FingerprintPath = Join-Path $ProjectRoot '.venv\requirements.sha256'
$RuntimeCheck = Join-Path $PSScriptRoot 'check_runtime.py'

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE. See the message above."
    }
}

try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        $PythonExecutable = $null
        $Launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($Launcher) {
            foreach ($Version in @('3.12', '3.13', '3.11')) {
                try {
                    $Candidate = & $Launcher.Source "-$Version" $RuntimeCheck '--print-executable' 2>$null
                } catch {
                    continue
                }
                if ($LASTEXITCODE -eq 0 -and $Candidate) {
                    $PythonExecutable = ($Candidate | Select-Object -Last 1).Trim()
                    break
                }
            }
        }
        if (-not $PythonExecutable) {
            $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($PythonCommand) {
                try {
                    $Candidate = & $PythonCommand.Source $RuntimeCheck '--print-executable' 2>$null
                } catch {
                    $Candidate = $null
                }
                if ($LASTEXITCODE -eq 0 -and $Candidate) {
                    $PythonExecutable = ($Candidate | Select-Object -Last 1).Trim()
                }
            }
        }
        if (-not $PythonExecutable) {
            throw 'Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/ with Tcl/Tk and the launcher enabled. Then run START_WINDOWS.bat again.'
        }
        Write-Host 'Creating an isolated Python environment...'
        Invoke-CheckedPython -Executable $PythonExecutable -Arguments @('-m', 'venv', '.venv')
    }
    Invoke-CheckedPython -Executable $VenvPython -Arguments @($RuntimeCheck)
    $Hash = (Get-FileHash -LiteralPath $Requirements -Algorithm SHA256).Hash
    $InstalledHash = if (Test-Path -LiteralPath $FingerprintPath) { (Get-Content -LiteralPath $FingerprintPath -Raw).Trim() } else { '' }
    if ($Hash -ne $InstalledHash) {
        Write-Host 'Installing tested dependencies. Internet is needed on the first run...'
        Invoke-CheckedPython -Executable $VenvPython -Arguments @('-m', 'pip', 'install', '--only-binary=:all:', '-r', $Requirements)
        Invoke-CheckedPython -Executable $VenvPython -Arguments @('-m', 'emotion_app', 'doctor')
        [System.IO.File]::WriteAllText($FingerprintPath, $Hash)
    }
    if ($Test) {
        Invoke-CheckedPython -Executable $VenvPython -Arguments @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
        Invoke-CheckedPython -Executable $VenvPython -Arguments @('-m', 'emotion_app', 'doctor')
        Write-Host 'Software tests passed. To check physical hardware, run: .venv\Scripts\python -m emotion_app doctor --camera 0'
    } else {
        Invoke-CheckedPython -Executable $VenvPython -Arguments @('-m', 'emotion_app', 'gui')
    }
    exit 0
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
