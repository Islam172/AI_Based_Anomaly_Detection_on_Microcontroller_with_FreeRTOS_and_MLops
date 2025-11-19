# Import the logging module for creating and managing log files
import logging

# Import the os module to interact with the operating system (e.g., file paths, directories)
import os

# Import the datetime module to work with dates and times
from datetime import datetime

# Create a log file name with the current timestamp
# The strftime method formats the datetime object into "month_day_year_hour_minute_second"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path to store log files
# os.getcwd() gets the current working directory
# os.path.join combines the working directory, "logs" folder, and log file name
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the "logs" directory (and any necessary parent directories) if it doesn't already exist
# exist_ok=True ensures that no error is raised if the directory already exists
os.makedirs(logs_path, exist_ok=True)

# Define the full path for the log file
# os.path.join combines the logs directory and the log file name
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specify the log file where logs will be saved
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Set the logging level to INFO (logs INFO, WARNING, ERROR, and CRITICAL)
)

# Explanation of the log format:
# - %(asctime)s: Timestamp of the log entry
# - %(lineno)d: Line number in the code where the log was generated
# - %(name)s: Name of the logger (default is 'root')
# - %(levelname)s: Log severity level (e.g., INFO, WARNING, ERROR)
# - %(message)s: The actual log message
