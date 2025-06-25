import os

# Define folder structure
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "scripts",
    "models",
    "src/config",
    "src/data",
    "src/features",
    "src/models",
    "src/utils",
    "tests"
]

# Define basic files to create
files = {
    "README.md": "# MLOps Project\n\nThis project implements a full machine learning pipeline.",
    ".gitignore": "*.pyc\n__pycache__/\n.env\n*.pkl\n",
    "requirements.txt": "# Add required Python packages here\n",
    "scripts/preprocess.py": "# Preprocessing logic\n",
    "scripts/train.py": "# Training logic\n",
    "scripts/evaluate.py": "# Evaluation logic\n",
    "scripts/predict.py": "# Prediction logic\n",
    "src/__init__.py": "",
    "tests/test_train.py": "# Write unit tests here\n",
    "src/config/config.yaml": "# Configuration parameters\n"
}

def create_structure(base_path="."):
    # Create folders
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

    # Create files
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"Created file: {full_path}")

if __name__ == "__main__":
    create_structure()
