# Create a virtual environment
python -m venv bhenv

# Activate the virtual environment
if ($IsWindows) {
    ./bhenv/Scripts/Activate.ps1
} else {
    ./bhenv/bin/Activate.ps1
}

# Install required packages
pip install -r requirements.txt
