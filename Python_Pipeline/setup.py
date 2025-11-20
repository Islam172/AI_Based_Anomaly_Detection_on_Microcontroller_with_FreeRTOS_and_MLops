'''
The setup.py file is an essential part of packaging and
distributing Python projects. It is used by setuptools
to define the configuration of your project, 
such as its metadata, dependencies, and more
'''



# Import necessary modules for creating a Python package
from setuptools import find_packages, setup

# Import List type hint from the typing module for specifying return types
from typing import List

# Define a function to read the requirements from a file and return them as a list of strings
def get_requirements() -> List[str]:
    """
    This function reads the 'requirements.txt' file and returns a list of requirements.
    Returns:
        List[str]: A list of package dependencies as strings.
    """
    # Initialize an empty list to store the requirements
    requirement_lst: List[str] = []

    try:
        # Open the 'requirements.txt' file in read mode
        with open('requirements.txt', 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

            # Process each line in the file
            for line in lines:
                # Remove leading and trailing whitespace
                requirement = line.strip()

                # Ignore empty lines and lines containing '-e .' (used for editable installations)
                if requirement and requirement != '-e .':
                    # Add valid requirements to the list
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        # Handle the case where 'requirements.txt' does not exist
        print("requirements.txt file not found")

    # Return the list of requirements
    return requirement_lst

# Define the setup function for packaging the project
setup(
    # The name of the package
    name="Anomaly Detection",

    # The version of the package
    version="0.0.1",

    # The author's name
    author="Islam Elmaaroufi",

    # Automatically find and include all packages in the project
    packages=find_packages(),

    # Install the dependencies listed in the 'requirements.txt' file
    install_requires=get_requirements()
)