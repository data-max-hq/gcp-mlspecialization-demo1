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
tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install Git
echo "Installing Git..."
sudo apt install -y git

# Install Pyenv
echo "Installing Pyenv..."
curl https://pyenv.run | bash

# Update shell configuration for Pyenv
echo "Configuring shell for Pyenv..."
echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc

# Restart the shell to load Pyenv
echo "Restarting shell to apply configuration..."
exec "$SHELL"

# Verify Pyenv installation
echo "Verifying Pyenv installation..."
pyenv --version

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

# Clone the GitHub repository
echo "Cloning the GitHub repository..."
gh repo clone data-max-hq/gcp-mlspecialization-demo1

# Navigate to the cloned repository
echo "Navigating to the project directory..."
cd gcp-mlspecialization-demo1

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
