from setuptools import find_packages, setup

name = 'Deepurify'
requires_list = open('./requirements.txt', 'r', encoding='utf8').readlines()
requires_list = [i.strip() for i in requires_list]

setup(
    name=name,
    version='1.1.0.1',
    author="Bohao Zou",
    author_email='csbhzou@comp.hkbu.edu.hk',
    description="The purification tool for improving the quality of MAGs.",
    python_requires=">=3.6",
    packages=find_packages(),
    package_data={"": ["*"]},
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["deepurify=Deepurify.cli:cli"]},

    install_requires=requires_list
)


# python -m twine upload dist/* -u csbhzou -p XJYzbh19960919
# python .\setup.py bdist_wheel
