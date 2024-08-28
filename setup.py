from setuptools import setup

setup(name='fiora',
      version='0.0.1',
      long_description='file: README.md',
      author='Yannek Nowatzky',
      license='MIT',
      classifiers=[
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],

      scripts=["scripts/fiora-predict"],
      packages=['fiora', 'fiora.GNN', 'fiora.IO',  'fiora.MOL', 'fiora.MS', 'fiora.visualization', 'models'],
      include_package_data=True,
      package_data={
        'models': ['fiora_OS_v0.1.0.pt', 'fiora_OS_v0.1.0_state.pt', 'fiora_OS_v0.1.0_params.json'],
      },
      install_requires=['numpy', 'seaborn', 'torch', 'torch_geometric', 'dill', 'rdkit', 'treelib', 'spectrum_utils', 'setuptools>=24.2.0'],
      python_requires='>=3.10.8',
      # Developers may also want to install: jupyter torchmetrics umap umap-learn pytest
      )
