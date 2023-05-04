from setuptools import find_packages,setup

def get_req(file_path):
    requirements=[]
    with open(r'requirements.txt','r') as f:
        req=[line.replace('\n','') for line in f.readlines()]
        if '-e .' in req:
            req.remove('-e .')
        return req



setup(
    name='sample_project',
    version='0.0.1',
    author_email='uppalurirakesh@gmail.com',
    author='Rakesh',
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)