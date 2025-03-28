from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT= '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n","") for req in requirements]


        if HYPEN_E_DOT in requirements:
            requirements = requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name="Predictive_Maintenance_RULPrediction",
    version = '0.0.1',
    author ='Kunal Lokhande',
    author_email = 'kunallokhandea99@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()
    )