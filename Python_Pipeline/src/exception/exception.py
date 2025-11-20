# Import the sys module to access system-specific parameters and functions
import sys

# Import a custom logger object from the networksecurity.logging module
from src.logging import logger

# Define a custom exception class for handling network security-related errors
class Exception(Exception):
    # Constructor to initialize the custom exception with additional details
    def __init__(self, error_message, error_details: sys):
        # Store the error message
        self.error_message = error_message
        
        # Extract traceback information using sys.exc_info()
        # exc_tb contains the traceback object
        _, _, exc_tb = error_details.exc_info()
        
        # Get the line number where the exception occurred
        self.lineno = exc_tb.tb_lineno
        
        # Get the file name where the exception occurred
        self.file_name = exc_tb.tb_frame.f_code.co_filename 
    
    # Override the __str__ method to provide a user-friendly string representation of the exception
    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )
        
# The main execution block of the script
if __name__ == '__main__':
    try:
        # Log an informational message indicating the start of the try block
        logger.logging.info("Enter the try block")
        
        # Intentionally cause a ZeroDivisionError by dividing 1 by 0
        a = 1 / 0
        
        # This line will never be executed because the exception halts execution
        print("This will not be printed", a)
    except Exception as e:
        # Catch any exception that occurs and raise a custom NetworkSecurityException
        # Pass the original exception message (e) and sys module for traceback details
        raise Exception(e, sys)
    
