import logging
# Configure logging
def setup_logging():
    """Configure logging for the application."""
    logger = logging.getLogger('beebot')
    logger.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    return logger

# Create logger instance
logger = setup_logging()