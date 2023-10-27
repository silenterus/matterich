Function GetPythonPath {
<#
    .SYNOPSIS
        This script checks various locations for a Python executable and returns the path to the executable if found.
    .DESCRIPTION
        The `GetPythonPath` function checks for Python executables using the following methods:

            1. Checks for the Python 'py' launcher, which should be present on most modern Python installations on Windows platforms.
               It prioritizes Python 3.x versions if multiple Python versions are found.

            2. Checks user-specified paths for Python executables. Paths for Python 3.x up to Python 3.11 are predefined.

            3. Checks a list of standard Python executable names, including different versions of Python 3.x.

        The function returns the first valid Python executable path found or throws an exception if no valid Python executable is found.
    .PARAMETER Version
        The version of Python executable to look for. Not mandatory.
    .PARAMETER PythonPath
        The path to the Python executable to look for. Not necessary if 'Version' is specified.
#>
    [CmdletBinding(PositionalBinding = $false)]
    param (
        [Parameter(Mandatory=$false)]
        [string][Alias('v')]$Version,

        [Parameter(Mandatory=$false)]
        [string][Alias('p')]$PythonPath
    )
    try {
        # First, check for the 'py' launcher to prioritize Python 3.x versions
        $PyLauncherPath = & py -0p 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pypath = ($PyLauncherPath -split "`n" | Select-String -Pattern '^\s*-V:\d+\.\d+\s+\*\s+').ToString().Trim().Split(' ')[-1]

            if (Test-Path $pypath) {
                Write-Host "Found Python over PyLauncher: ${pypath}"
                return $pypath
            }
        }

        # Next, check the paths specified by the user, if any
        $PythonExecPaths = @("C:\\Python3\\python.exe","C:\\Python310\\python.exe", "C:\\Python311\\python.exe","C:\\python3\\python.exe","C:\\python310\\python.exe", "C:\\python311\\python.exe")  # Add user-specified paths here
        foreach ($Path in $PythonExecPaths) {
            if (Test-Path $Path) {
                return $Path
            }
        }

        # Finally, check a list of Python executable names
        $PythonExecNames = @("python", "python310", "python311", "python3")
        foreach ($ExecName in $PythonExecNames) {
            $PythonExecPath = (Get-Command -Name $ExecName -ErrorAction SilentlyContinue).Path
            if ($PythonExecPath -ne $null) {
                return $PythonExecPath
            }
        }

        # If no valid executable paths were found, throw an exception
        throw "No valid Python executable found."
    } catch {
        Write-Host "An error occurred: $_"
        return $null
    }
}



$global:PossibleGitUrls = @(
                            ""
                            ""
)

Function DownloadGit {
    try {
        $url = "https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/Git-2.33.0.2-64-bit.exe"
        $output = "Git-Installer.exe"
        Invoke-WebRequest -Uri $url -OutFile $output
        Write-Host "Git installer downloaded."
    } catch [System.Net.WebException] {
        Write-Host "Failed to download Git. Please check your internet connection."
    } catch {
        Write-Host "An unexpected error occurred."
    }
}

Function DownloadGit2 {
    try {
        $url = "https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/Git-2.33.0.2-64-bit.exe"
        $output = "./GitInstaller2.exe"
        Invoke-WebRequest -Uri $url -OutFile $output
    } catch {}
}

Export-ModuleMember -Function DownloadGit

DownloadGit2
DownloadGit

Write-Host ""
