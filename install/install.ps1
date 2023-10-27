Function GetPythonPath {
    param (
        [Parameter(Mandatory=$false)]
        [string][Alias('v')]$Version,
        [Parameter(Mandatory=$false)]
        [string][Alias('p')]$PythonPath
    )
    try {
        $PythonExecPaths = @("C:\\Python3\\python.exe", "C:\\Python310\\python.exe", "C:\\Python311\\python.exe")
        foreach ($Path in $PythonExecPaths) {
            if (Test-Path $Path) {
                return $Path
            }
        }
        foreach ($ExecName in @("python", "python310", "python311", "python3")) {
            $PythonExecPath = (Get-Command -Name $ExecName -ErrorAction SilentlyContinue).Path
            if ($PythonExecPath -ne $null) {
                return $PythonExecPath
            }
        }
        throw "No valid Python executable found."
    } catch {
        Write-Host "An error occurred: $_"
        return $null
    }
}

Function ManageVenv {
    param (
        [string]$PythonPath
    )
    try {
        $venvPaths = @(".\venv", "..\venv")
        foreach ($venv in $venvPaths) {
            if (Test-Path $venv) {
                Remove-Item -Recurse -Force $venv -ErrorAction SilentlyContinue
                Write-Host "Old virtual environment at $venv removed."
            }
        }
        
        & $PythonPath -m venv ".\venv"

        $activationScripts = @(".\venv\Scripts\activate", ".\venv\Scripts\activate.ps1", ".\venv\Scripts\activate.bat")
        $activated = $false
        foreach ($script in $activationScripts) {
            if (Test-Path $script) {
                & $script
                $activated = $true
                break
            }
        }

        if (-not $activated) {
            throw "Failed to activate virtual environment."
        }

        Write-Host "Virtual environment activated successfully."
        & pip install gradio
        & pip install metaseg
        & pip install -r requirements.txt -r req.txt -r requirements-dev.txt
    } catch {
        Write-Host "An error occurred: $_"
    }
}

try {
    $pathPython = GetPythonPath
    if ($pathPython -ne $null) {
        ManageVenv -PythonPath $pathPython
    } else {
        throw "No valid Python executable found."
    }
} catch {
    Write-Host "An error occurred: $_"
}
