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
          'holoviews==1.14.8',
          'bokeh==2.4.2',
          'panel==0.12.7',
          'param==1.12.1',
          'xarray==0.20.2',
          'datashader==0.13.0',
          'pyvista',
          'pandas==1.3.5',
          'numpy==1.18.5',
          'scipy==1.7.3',
          'scikit-image==0.19.2',
          'tqdm==4.64.0',
          'scikit-learn==1.0.2',
          'imagecodecs==2021.11.20',
          'pytest==7.1.1',
          'luigi==3.0.3',
          #'dl-utils @ git+https://github.com/fmi-basel/dl-utils',
          'improc @ git+https://github.com/fmi-basel/improc',
          'inter_view @ git+https://github.com/fmi-basel/inter-view',
          'flowdec==1.1.0'
      ],
      zip_safe=False)
