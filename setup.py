from setuptools import find_packages,setup

from typing import List

REQUIREMENTS_FILE="requirements.txt"
HYPHEN="-e ."

def get_requirements()->List[str]:
    with open(REQUIREMENTS_FILE) as req_file:
        req_list=req_file.readlines()
    req_list=[req_name.replace("\n","") for req_name in req_list]
    if HYPHEN in req_list:
        req_list.remove(HYPHEN)
    return req_list



setup(
    name="credit card default",
    version="0.0.1",
    author="tuerydany",
    authormail="tureydany19@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)