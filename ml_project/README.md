## Run training pipeline mode model 
```python -m src.pipeline.training_pipeline```

## Run prediction pipeline model 
```python -m src.pipeline.prediction_pipeline```


## Use mlflow:
```mlflow ui```


## Setup.py file for versioning

```python
from pathlib import Path

def generate_setup_py():
    name = input("Project name: ").strip()
    version = input("Version: ").strip()
    description = input("Description: ").strip()
    author = input("Author: ").strip()
    email = input("Author email: ").strip()

    content = f'''
from setuptools import setup, find_packages

setup(
    name="{name}",
    version="{version}",
    description="{description}",
    author="{author}",
    author_email="{email}",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    install_requires=["pandas", "scikit-learn", "joblib", "mlflow"],
)
'''

    Path("setup.py").write_text(content)

    print("setup.py generated!")

if __name__ == "__main__":
    
    generate_setup_py()





