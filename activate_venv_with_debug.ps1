# Activate the virtual environment and enable LEGION payload debug
# Usage: .\activate_venv_with_debug.ps1

# Activate the venv
& "${PWD}\.venv\Scripts\Activate.ps1"

# Set debug flag for this session
$env:LEGION_PAYLOAD_DEBUG = 'true'
Write-Host "LEGION_PAYLOAD_DEBUG set to: $env:LEGION_PAYLOAD_DEBUG"
