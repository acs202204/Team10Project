
#!/bin/bash

# -----------------------------
# CONFIGURATION
# -----------------------------
PROJECT_DIR="/home/team10sp/AI_Work/HandTracking/venv"
ENV_DIR="$PROJECT_DIR/venv"

# -----------------------------
# MOVE TO PROJECT DIRECTORY
# -----------------------------
cd "$PROJECT_DIR" || exit

# -----------------------------
# CREATE OR ACTIVATE VENV
# -----------------------------
if [ -d "$ENV_DIR" ]; then
    echo "Virtual environment already exists at: $ENV_DIR"
else
    echo "Creating virtual environment at: $ENV_DIR"
    python3 -m venv "$ENV_DIR"
fi

# Activate environment
source "$ENV_DIR/bin/activate"

echo "Moving to virtual environment.."
cd /home/team10sp/AI_Work/HandTracking/venv


# -----------------------------
# FUNCTION TO CHECK A PYTHON PACKAGE
# -----------------------------
check_pkg() {
    pkg="$1"
    import_name="$2"

    python3 -c "import $import_name" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Package '$pkg' is NOT installed."
        read -p "Install $pkg now (y/n): " choice
        if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
            pip install "$pkg"
        else
            echo "Skipping installation of $pkg"
        fi
    else
        echo "Package '$pkg' is already installed."
    fi
}

# -----------------------------
# CHECK REQUIRED PACKAGES
# -----------------------------
check_pkg "opencv-python" "cv2"
check_pkg "torch" "torch"
check_pkg "ultralytics" "ultralytics"

echo "Environment ready."

