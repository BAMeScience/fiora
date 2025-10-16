from pathlib import Path
from setuptools import setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name='fiora',
    version='1.0.1',
    description='In silico fragmentation and MS/MS simulation',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Yannek Nowatzky',
    license='MIT',
    classifiers=[
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    scripts=["scripts/fiora-predict"],
    packages=['fiora', 'fiora.GNN', 'fiora.IO', 'fiora.MOL', 'fiora.MS', 'fiora.visualization', 'models'],
    include_package_data=True,
    package_data={
        'models': [
            'fiora_OS_v0.1.0.pt',
            'fiora_OS_v0.1.0_state.pt',
            'fiora_OS_v0.1.0_params.json',
            'fiora_OS_v1.0.0.pt',
            'fiora_OS_v1.0.0_state.pt',
            'fiora_OS_v1.0.0_params.json',
        ],
    },
    install_requires=['numpy', 'seaborn', 'torch', 'torch_geometric>=2.6,<2.7', 'dill', 'rdkit', 'treelib', 'spectrum_utils', 'ipython>=8'],
    python_requires='>=3.10.8',
)