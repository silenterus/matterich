Function GetPythonPath {
    [CmdletBinding(PositionalBinding = $false)]
    param (
        [Parameter(Mandatory=$false)]
        [string][Alias('v')]$Version,

        [Parameter(Mandatory=$false)]
        [string][Alias('p')]$PythonPath
    )
    try {
        $PyLauncherPath = & py -0p 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pypath = ($PyLauncherPath -split "`n" | Select-String -Pattern '^\s*-V:\d+\.\d+\s+\*\s+').ToString().Trim().Split(' ')[-1]
            if (Test-Path $pypath) {
                return $pypath
            }
        }

        $PythonExecPaths = @("C:\\Python3\\python.exe","C:\\Python310\\python.exe", "C:\\Python311\\python.exe")
        foreach ($Path in $PythonExecPaths) {
            if (Test-Path $Path) {
                return $Path
            }
        }

        $PythonExecNames = @("python", "python310", "python311", "python3")
        foreach ($ExecName in $PythonExecNames) {
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

Function DownloadGit {
    try {
        $url = "https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/Git-2.33.0.2-64-bit.exe"
        $output = "Git-Installer.exe"
        Invoke-WebRequest -Uri $url -OutFile $output
    } catch [System.Net.WebException] {
        Write-Host "Failed to download Git. Please check your internet connection."
    } catch {
        Write-Host "An unexpected error occurred."
    }
}

try {
    $hasCreatedEnv = $false
    $pathPython = GetPythonPath
    if ($pathPython -ne $null) {
        & $pathPython -m venv venv
        if (Test-Path ".\venv\Scripts\activate") {
            . .\venv\Scripts\activate
            Write-Host "Virtual environment activated successfully."
            $hasCreatedEnv = $true
        }
        & $pathPython -m venv venv
        elseif (Test-Path ".\venv\Scripts\activate.ps1") {
            & .\venv\Scripts\activate.ps1
            Write-Host "Virtual environment activated successfully."
            $hasCreatedEnv = $true

        }
        elseif (Test-Path ".\venv\Scripts\activate.bat") {
            & .\venv\Scripts\activate.bat
            Write-Host "Virtual environment activated successfully."
            $hasCreatedEnv = $true

        }
        if (Test-Path "..\venv\Scripts\activate") {
            . ..\venv\Scripts\activate
            Write-Host "Virtual environment activated successfully."
            $hasCreatedEnv = $true

        }
        & $pathPython -m venv venv
        elseif (Test-Path "..\venv\Scripts\activate.ps1") {
            & ..\venv\Scripts\activate.ps1
            Write-Host "Virtual environment activated successfully."
            $hasCreatedEnv = $true

        }
        elseif (Test-Path "..\venv\Scripts\activate.bat") {
            & ..\venv\Scripts\activate.bat
            Write-Host "Virtual environment activated successfully."
            $hasCreatedEnv = $true

        }
        else {
            throw "Failed to activate virtual environment."
        }


        if($hasCreatedEnv){


        }
    } else {
        throw "No valid Python executable found."
    }
} catch {
    Write-Host "An error occurred: $_"
}
