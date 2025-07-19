import os
import pathlib
from setuptools import setup, find_packages

# 获取项目根目录路径
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

def load_requirements(path_dir=here, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [line.strip() for line in file.readlines()]
    requirements = []
    for line in lines:
        if comment_char in line:
            line = line[: line.index(comment_char)]
        if line: 
            requirements.append(line)
    return requirements

setup(
    name='soptx',
    version='1.0.0',
    description='Structural Optimization Topology Simulation Software',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brighthe/soptx',
    author='Liang He',
    author_email='',
    license="GNU",
    packages=find_packages(),
    install_requires=load_requirements(),
    python_requires=">=3.10",
)
