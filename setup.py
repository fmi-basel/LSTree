from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz', 'Gustavo de Medeiros'
]

# setup.
setup(name='LSTree',
      version='0.1',
      description='Utils to process lightsheet movies of organoids',
      author=', '.join(contrib),
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'pyviz',
          'holoviews',
          'bokeh',
          'panel',
          'param',
          'xarray',
          'datashader',
          'pytables',
          'pyvista',
          'pandas',
          'numpy',
          'scipy',
          'scikit-image',
          'tqdm',
          'scikit-learn',
          'imagecodecs',
          'pytest',
          'luigi',
          'dl-utils @ git+https://github.com/fmi-basel/dl-utils',
          'improc @ git+https://github.com/fmi-basel/improc',
          'inter_view @ git+https://github.com/fmi-basel/inter-view',
          'flowdec',
      ],
      zip_safe=False)
