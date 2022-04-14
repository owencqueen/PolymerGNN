from distutils.core import setup

#reqs = open('requirements.txt', 'r').readlines()

setup(
    name='polymerlearn',
    version='0.0.1',
    author = 'Owen Queen',
    autor_email = 'oqueen@vols.utk.edu',
    description='Machine learning for polymers',
    #install_requires = reqs,
    packages=['polymerlearn'],
    zip_safe=False
)