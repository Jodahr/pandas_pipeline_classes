from setuptools import setup

setup(name='pandas_pipeline_classes',
      version='0.2',
      description='pipeline classes which return pandas DataFrames',
      url='http://githubcom/jodahr/pandas_pipeline_classes',
      author='jodahr',
      author_email='marcelrothering@fastmail.fm',
      licence='MIT',
      packages=['pandas_pipeline_classes'],
      install_requires = ['pandas', 'sklearn', 'numpy']
      zip_safe=False)
