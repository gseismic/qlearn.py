from setuptools import setup, find_packages


setup(
    name='qlearn', 
    version='0.0.1', 
    packages=find_packages(),
    description='Collections on Reinforcement Learning'],
    install_requires = ['torch', 'numpy', 'loguru'],
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/chatgpt_3k',
    zip_safe=False,
    author_email='liushengli203@163.com'
)
