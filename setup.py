from setuptools import setup
import glob

setup(name='deforest',
      packages = ['deforest'],
      version='0.1',
      description='Deforest, prototype.',
      data_files=[('./cfg/',glob.glob('./cfg/*'))],
      url='https://bitbucket.org/sambowers/deforest',
      author='Samuel Bowers',
      author_email='sam.bowers@ed.ac.uk',
      license='GNU General Public License',
      zip_safe=False)

