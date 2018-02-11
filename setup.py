import re

from setuptools import setup

with open('moisan2011.py', 'r', encoding='utf-8') as f:
    lines = f.read()
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        lines, re.MULTILINE).group(1)
    description = re.search(r'^u\"\"\"(.*)',
                            lines, re.MULTILINE).group(1)
    long_description = re.search(r'^u\"\"\"(.*)^\"\"\"',
                                 lines, re.MULTILINE | re.DOTALL).group(1)
    author = re.search(r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]',
                       lines, re.MULTILINE).group(1)

setup(
    name='moisan2011',
    version=version,
    description=description,
    long_description=long_description,
    url='',
    author=author,
    author_email='',
    py_modules=['moisan2011'],
    license='BSD-3',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Software Development :: Build Tools',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering'],
    test_suite='setup.my_test_suite',
    install_requires=['numpy'],
)
