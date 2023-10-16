from setuptools import setup

setup(name='Fiora',
      version='0.0.1',
      long_description='file: README.md',
      author='Yannek Nowatzky',
      license='MIT',
      classifiers=[
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],
      # entry_points={
      #   "console_scripts": [
      #       "fiora-predict = scripts.predict:main",
      #   ],
      #   },
      scripts=["scripts/fiora-predict"],
      include_package_data=True,
      packages=['fiora', 'fiora.GNN', 'fiora.IO',  'fiora.MOL', 'fiora.MS', 'fiora.visualization'],
      install_requires=['numpy', 'seaborn', 'torch', 'torch_geometric', 'dill', 'rdkit', 'treelib', 'spectrum_utils', 'setuptools>=24.2.0'],
      python_requires='>=3.10.8',
      # Developers may also want to install: jupyter torchmetrics umap umap-learn
      )
