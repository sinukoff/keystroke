from setuptools import setup, find_packages

reqs = []
for line in open('requirements.txt', 'r').readlines():
    reqs.append(line)

setup(
    name='keystroke',
    version='0.1',
    description='Identify online users by their keystroke patterns',
    author='Evan Sinukoff',
    author_email='sinukoej@gmail.com',
    packages=find_packages(),
    install_requires=reqs,
    entry_points={'console_scripts': ['keystroke=keystroke.cli:main']},
    include_package_data=True
)


