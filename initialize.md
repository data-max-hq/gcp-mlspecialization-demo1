# Script Initialization Guide

This guide outlines the steps required to initialize the script for the GCP ML Specialization Demo.

## Prerequisites

Ensure that you have `sudo` privileges on your system.

## Steps

1. **Update package lists:**
    ```sh
    sudo apt update
    ```

2. **Install essential build tools and dependencies:**
    ```sh
    sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

3. **Install Pyenv:**
    ```sh
    curl https://pyenv.run | bash
    ```

4. **Update shell configuration:**
    ```sh
    echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
    ```

5. **Restart the shell:**
    ```sh
    exec "$SHELL"
    ```

6. **Verify Pyenv installation:**
    ```sh
    pyenv --version
    ```

7. **Install Python 3.10.12:**
    ```sh
    pyenv install 3.10.12
    ```

8. **Set Python 3.10.12 as the global version:**
    ```sh
    pyenv global 3.10.12
    ```

9. **Install Git:**
    ```sh
    sudo apt install git
    ```

10. **Set up a virtual environment:**
    ```sh
    python3 -m venv venv
    ```

11. **Activate the virtual environment:**
    ```sh
    source venv/bin/activate
    ```

12. **Clone the GitHub repository:**
    ```sh
    gh repo clone data-max-hq/gcp-mlspecialization-demo1
    ```

13. **Clone the GitHub repository:**
    ```sh
    cd gcp-mlspecialization-demo1
    ```

14. **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

15. **Navigate to the project directory:**
    ```sh
    cd chicago_taxi_pipeline
    ```

16. **Run the pipeline script:**
    ```sh
    python -m pipeline.run_pipeline
    ```

17. **Submit the pipeline script:**
    ```sh
    python -m pipeline.submit_pipeline
    ```

## Conclusion

Following these steps will set up the environment and run the required scripts for the GCP ML Specialization Demo. If you encounter any issues, please refer to the repository's documentation.
