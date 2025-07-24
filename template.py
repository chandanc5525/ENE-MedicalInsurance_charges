import os
from pathlib import Path

project_name = input("Enter your project name: ")

def create_project_structure(project_name: str):
    folders = [
        f"{project_name}/data",
        f"{project_name}/artifacts",
        f"{project_name}/mlruns",
        f"{project_name}/src/pipeline",
        f"{project_name}/src/components",
        f"{project_name}/src/utils",
    ]

    files = {
    f"{project_name}/data/heart-disease.csv": "",
    f"{project_name}/artifacts/model.joblib": "",
    f"{project_name}/src/pipeline/training_pipeline.py": "",
    f"{project_name}/src/pipeline/prediction_pipeline.py": "",
    f"{project_name}/src/components/data_loader.py": "",
    f"{project_name}/src/components/trainer.py": "",
    f"{project_name}/src/components/evaluator.py": "",
    f"{project_name}/src/utils/logger.py": "",
    f"{project_name}/.gitignore": "*.pyc\n__pycache__/\nmlruns/\nartifacts/\n",
    f"{project_name}/README.md": f"# {project_name}\n\nStandard MLOps project structure.",
    }


from pathlib import Path

def create_project_structure(project_name: str):
    structure = [
        (f"{project_name}/data", None),
        (f"{project_name}/artifacts", None),
        (f"{project_name}/mlruns", None),
        (f"{project_name}/src/pipeline", None),
        (f"{project_name}/src/components", None),
        (f"{project_name}/src/utils", None),

        (f"{project_name}/data/heart-disease.csv", ""),
        (f"{project_name}/artifacts/model.joblib", ""),
        (f"{project_name}/src/pipeline/training_pipeline.py", ""),
        (f"{project_name}/src/pipeline/prediction_pipeline.py", ""),
        (f"{project_name}/src/components/data_loader.py", ""),
        (f"{project_name}/src/components/trainer.py", ""),
        (f"{project_name}/src/components/evaluator.py", ""),
        (f"{project_name}/src/utils/logger.py", ""),
        (f"{project_name}/.gitignore", ""),
        (f"{project_name}/README.md", ""),
    ]

    for path_str, content in structure:
        path = Path(path_str)
        if content is None:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    print(f"Project '{project_name}' structure created.")


if __name__ == "__main__":
    project_name = input("Enter your project name: ").strip()
    if project_name:
        create_project_structure(project_name)
    else:
        print("Project name cannot be empty.")

import os
from pathlib import Path

project_name = input("Enter your project name: ")

def create_project_structure(project_name: str):
    folders = [
        f"{project_name}/data",
        f"{project_name}/artifacts",
        f"{project_name}/mlruns",
        f"{project_name}/src/pipeline",
        f"{project_name}/src/components",
        f"{project_name}/src/utils",
    ]

    files = {
    f"{project_name}/data/heart-disease.csv": "",
    f"{project_name}/artifacts/model.joblib": "",
    f"{project_name}/src/pipeline/training_pipeline.py": "",
    f"{project_name}/src/pipeline/prediction_pipeline.py": "",
    f"{project_name}/src/components/data_loader.py": "",
    f"{project_name}/src/components/trainer.py": "",
    f"{project_name}/src/components/evaluator.py": "",
    f"{project_name}/src/utils/logger.py": "",
    f"{project_name}/.gitignore": "*.pyc\n__pycache__/\nmlruns/\nartifacts/\n",
    f"{project_name}/README.md": f"# {project_name}\n\nStandard MLOps project structure.",
    }


from pathlib import Path

def create_project_structure(project_name: str):
    structure = [
        (f"{project_name}/data", None),
        (f"{project_name}/artifacts", None),
        (f"{project_name}/mlruns", None),
        (f"{project_name}/src/pipeline", None),
        (f"{project_name}/src/components", None),
        (f"{project_name}/src/utils", None),

        (f"{project_name}/data/heart-disease.csv", ""),
        (f"{project_name}/artifacts/model.joblib", ""),
        (f"{project_name}/src/pipeline/training_pipeline.py", ""),
        (f"{project_name}/src/pipeline/prediction_pipeline.py", ""),
        (f"{project_name}/src/components/data_loader.py", ""),
        (f"{project_name}/src/components/trainer.py", ""),
        (f"{project_name}/src/components/evaluator.py", ""),
        (f"{project_name}/src/utils/logger.py", ""),
        (f"{project_name}/.gitignore", ""),
        (f"{project_name}/README.md", ""),
    ]

    for path_str, content in structure:
        path = Path(path_str)
        if content is None:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    print(f"Project '{project_name}' structure created.")


if __name__ == "__main__":
    project_name = input("Enter your project name: ").strip()
    if project_name:
        create_project_structure(project_name)
    else:
        print("Project name cannot be empty.")