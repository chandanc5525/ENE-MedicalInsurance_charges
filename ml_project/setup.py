
from setuptools import setup, find_packages

setup(
    name="Medical Insurance Charges Prediction Model",
    version="1.0.0",
    description="This is Multivariate/Regression Analysis Problem",
    author="Chandan Chaudhari",
    author_email="chaudhari.chandan22@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy","pandas", "scikit-learn", "joblib", "mlflow","matplotlib","seaborn","loguru"],
)
