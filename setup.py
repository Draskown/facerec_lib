from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="manager_face_recognizer",
    version="0.0.2",
    description="A face recognition script for the nun robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["mngr_facerec"],
    keywords="face-recognition, AI, computer-vision",
    install_requires=[
        "face-recognition==1.3.0",
        "face-recognition-models==0.3.0",
        "opencv-python==4.7.0.72",
    ],
    setup_requires=["flake8"],
)
