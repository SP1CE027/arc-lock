# ============================================================
# ArcLock Setup
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host "          ArcLock Setup"
Write-Host "========================================"
Write-Host ""


# ============================================================
# PROJECT DIRECTORY
# ============================================================

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $ProjectRoot

Write-Host "[1/7] Project directory:"
Write-Host "       $ProjectRoot"
Write-Host ""


# ============================================================
# CHECK PYTHON
# ============================================================

Write-Host "[2/7] Checking Python..."

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue

if (-not $PythonCommand) {

    Write-Host ""
    Write-Host "ERROR: Python was not found."
    Write-Host ""
    Write-Host "Install Python 3.11 or newer and make sure"
    Write-Host "Python is available from the command line."
    Write-Host ""

    exit 1
}

$PythonVersion = python --version

Write-Host "       $PythonVersion"
Write-Host ""


# ============================================================
# CREATE VIRTUAL ENVIRONMENT
# ============================================================

Write-Host "[3/7] Creating virtual environment..."

if (-not (Test-Path ".venv\Scripts\python.exe")) {

    python -m venv .venv

    if ($LASTEXITCODE -ne 0) {

        Write-Host ""
        Write-Host "ERROR: Failed to create virtual environment."
        exit 1
    }

    Write-Host "       Virtual environment created."

}
else {

    Write-Host "       Virtual environment already exists."
}

Write-Host ""


# ============================================================
# PYTHON PATH
# ============================================================

$VenvPython = Join-Path `
    $ProjectRoot `
    ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {

    Write-Host ""
    Write-Host "ERROR: Virtual environment Python was not found."
    exit 1
}


# ============================================================
# UPGRADE PIP
# ============================================================

Write-Host "[4/7] Updating pip..."

& $VenvPython -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: Failed to update pip."
    exit 1
}

Write-Host ""


# ============================================================
# INSTALL DEPENDENCIES
# ============================================================

Write-Host "[5/7] Installing dependencies..."

if (-not (Test-Path "requirements.txt")) {

    Write-Host ""
    Write-Host "ERROR: requirements.txt was not found."
    exit 1
}

& $VenvPython -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: Dependency installation failed."
    exit 1
}

Write-Host ""


# ============================================================
# CREATE DIRECTORIES
# ============================================================

Write-Host "[6/7] Creating required directories..."

if (-not (Test-Path "data")) {

    New-Item `
        -ItemType Directory `
        -Path "data" `
        | Out-Null
}

if (-not (Test-Path "logs")) {

    New-Item `
        -ItemType Directory `
        -Path "logs" `
        | Out-Null
}

Write-Host "       data\"
Write-Host "       logs\"
Write-Host ""


# ============================================================
# CHECK EMBEDDING
# ============================================================

Write-Host "[7/7] Checking biometric embedding..."

$EmbeddingFile = Join-Path `
    $ProjectRoot `
    "data\yash_embeddings.npy"

if (Test-Path $EmbeddingFile) {

    Write-Host "       Embedding found."

}
else {

    Write-Host ""
    Write-Host "WARNING: yash_embeddings.npy was not found."
    Write-Host ""
    Write-Host "Copy your backed-up embedding to:"
    Write-Host ""
    Write-Host "       $EmbeddingFile"
    Write-Host ""
}


# ============================================================
# CREATE WINDOWS STARTUP SHORTCUT
# ============================================================

Write-Host ""
Write-Host "Creating Windows Startup shortcut..."

$StartupFolder = [Environment]::GetFolderPath(
    "Startup"
)

$ShortcutPath = Join-Path `
    $StartupFolder `
    "ArcLock.lnk"

$PythonW = Join-Path `
    $ProjectRoot `
    ".venv\Scripts\pythonw.exe"

$MainPy = Join-Path `
    $ProjectRoot `
    "main.py"

if (-not (Test-Path $PythonW)) {

    Write-Host ""
    Write-Host "WARNING: pythonw.exe was not found."
    Write-Host "Startup shortcut was not created."

}
elseif (-not (Test-Path $MainPy)) {

    Write-Host ""
    Write-Host "WARNING: main.py was not found."
    Write-Host "Startup shortcut was not created."

}
else {

    $WshShell = New-Object -ComObject WScript.Shell

    $Shortcut = $WshShell.CreateShortcut(
        $ShortcutPath
    )

    $Shortcut.TargetPath = $PythonW

    $Shortcut.Arguments = "`"$MainPy`""

    $Shortcut.WorkingDirectory = $ProjectRoot

    $Shortcut.Description = "ArcLock"

    $Shortcut.Save()

    Write-Host "       Startup shortcut created."
}


# ============================================================
# IMPORT TEST
# ============================================================

Write-Host ""
Write-Host "Running dependency test..."

& $VenvPython -c "import cv2, numpy, insightface, onnxruntime, pystray, PIL; print('All required Python packages imported successfully.')"

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: Dependency import test failed."
    exit 1
}


# ============================================================
# COMPLETE
# ============================================================

Write-Host ""
Write-Host "========================================"
Write-Host "       ArcLock setup complete"
Write-Host "========================================"
Write-Host ""

if (Test-Path $EmbeddingFile) {

    Write-Host "Embedding:      FOUND"

}
else {

    Write-Host "Embedding:      MISSING"

}

Write-Host "Virtual env:    READY"
Write-Host "Dependencies:   READY"
Write-Host "Startup:        CONFIGURED"
Write-Host ""

Write-Host "To start ArcLock manually:"
Write-Host ""

Write-Host "    .\.venv\Scripts\pythonw.exe main.py"

Write-Host ""
Write-Host "ArcLock will also start automatically when you log into Windows."
Write-Host ""