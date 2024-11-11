#!/bin/bash

# Script Initialization for GCP ML Specialization Demo

# Check for sudo privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo privileges."
  exit 1
fi

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install essential build tools and dependencies
echo "Installing essential build tools and dependencies..."
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

# Install Pyenv
echo "Installing Pyenv..."
curl https://pyenv.run | bash

# Update shell configuration for Pyenv
echo "Configuring shell for Pyenv..."
{
  echo 'export PYENV_ROOT="$HOME/.pyenv"'
  echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
  echo 'eval "$(pyenv init --path)"'
  echo 'eval "$(pyenv init -)"'
} >> ~/.bashrc

# Source the updated bashrc to apply Pyenv path and initialize configuration
source ~/.bashrc

# Verify Pyenv installation
echo "Verifying Pyenv installation..."
if ! command -v pyenv > /dev/null; then
  echo "Pyenv installation failed. Please check the installation steps."
  exit 1
fi

# Install Python 3.10.12
echo "Installing Python 3.10.12 via Pyenv..."
pyenv install 3.10.12

# Set Python 3.10.12 as the global version
echo "Setting Python 3.10.12 as the global Python version..."
pyenv global 3.10.12

# Set up a virtual environment
echo "Setting up a Python virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Navigate to the Chicago taxi pipeline directory
echo "Navigating to the Chicago taxi pipeline directory..."
cd chicago_taxi_pipeline

# Run the pipeline script
echo "Running the pipeline script..."
python -m pipeline.run_pipeline

# Submit the pipeline script
echo "Submitting the pipeline script..."
python -m pipeline.submit_pipeline

echo "Setup and execution completed for GCP ML Specialization Demo."
