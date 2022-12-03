from setuptools import setup, find_packages

setup(name='molSimplify',
      version='v1.7.1',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'molsimplify = molSimplify.__main__:main',
              'molscontrol = molSimplify.molscontrol.molscontrol:main',
              'jobmanager = molSimplify.job_manager.resub:main']
      },
      package_dir={'molSimplify': 'molSimplify'},
      package_data={
          'molSimplify': ['Data/*.dat', 'Bind/*.dat', 'Ligands/*.dict',
                          'icons/*.png', 'python_nn/*.csv', 'python_krr/*.csv',
                          'tf_nn/*/*', 'molscontrol/*/*']
      },
      data_files=[('molSimplify', ['molSimplify/Data/ML.dat'])],
      install_requires=['openbabel', 'numpy', 'scipy', 'scikit-learn',
                        'keras', 'tensorflow', 'pyyaml'],
      setup_requires=['pytest-runner'],  # this may result some package conflict in local conda build. comment it out if needed.
      tests_require=['pytest'],
      include_package_data=True
      )
