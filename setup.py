import os
from setuptools import setup


def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='dts',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='Distributed Time Series Modeling with Apache Spark',
      keywords='spark, spark-ml, pyspark, mapreduce',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='https://github.com/feng-li/dts',
      author='Feng Li',
      author_email='feng.li@cufe.edu.cn',
      license='MIT',
      packages=['dts'],
      install_requires=[
          'pyspark >= 3.1.1',
          'pyarrow >= 0.15.0',
          'sklearn >= 0.21.2',
          'numpy   >= 1.16.3',
          'pandas  >= 0.23.4',
      ],
      zip_safe=False,
      python_requires='>=3.7',
)
