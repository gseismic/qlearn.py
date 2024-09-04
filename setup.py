from setuptools import setup, find_packages


setup(
    name='rlearn', 
    version='0.0.2', 
    packages=find_packages(),
    description='Reinforcement Learning Algorithms',
    install_requires = ['torch', 'numpy', 'loguru'],
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/rlearn.py',
    zip_safe=False,
    author_email='liushengli203@163.com'
)
