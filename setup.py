import os
from setuptools import setup


def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()


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
          'sklearn >= 0.21.2',
          'numpy   >= 1.16.3',
          'pandas  >= 0.23.4',
          'statsmodels >= 0.13.0',
          'autograd >= 1.5',
          'numdifftools >=0.9.40',
          'scipy >= 1.9.2',
          'pandas >= 1.4.0',
      ],
      zip_safe=False,
      python_requires='>=3.7')
