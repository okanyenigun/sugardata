from setuptools import setup, find_packages


with open("Readme.md", "r") as f:
    long_description = f.read()


setup(
    name="sugardata",
    version="0.0.3",
    description="Generates synthetic datasets tailored for transformer-based models",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/okanyenigun/sugardata",
    author="Okan Yenigün",
    author_email="okanyenigun@gmail.com",
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Documentation',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'transformers==4.48.0',
        'pandas==2.2.3',
        'torch==2.5.1',
        'ipywidgets==8.1.5',
        'langchain==0.3.18',
        'langchain-openai==0.3.5',
        'langchain-google-genai==2.0.9',
        'langchain-together==0.3.0',
        'langchain-groq==0.2.4',
        'langchain-ollama==0.2.3',
        'datasets==3.2.0',
        'deep-translator==1.11.4',
    ],
    python_requires='>=3.7',
)