import os
import os.path

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """ Find the version"""
    version_file = os.path.join(BASEDIR, 'ovos_m2v_pipeline', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if int(alpha):
        version += f"a{alpha}"
    return version


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


PLUGIN_ENTRY_POINT = 'ovos-m2v-pipeline=ovos_m2v_pipeline:Model2VecIntentPipeline'

setup(
    name="ovos_m2v_pipeline",
    version=get_version(),
    author="JarbasAI",
    description="Model2Vec intent engine for OVOS",
    long_description="",
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    keywords="natural language processing",
    entry_points={'opm.pipeline': PLUGIN_ENTRY_POINT},
    url="https://github.com/TigreGotico/ovos-m2v-pipeline",
    packages=["ovos_m2v_pipeline"],
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License'
    ],
    install_requires=required('requirements.txt')
)
