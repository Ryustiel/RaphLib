
__author__ = "Raphael Nguyen"
__copyright__ = "© 2025 Raphael Nguyen"
__license__ = "MIT"
__version__ = "1.0.0"

success = False
try:
    from src import *
except ModuleNotFoundError as e:
    try:
        from .src import *
        success = True
    except Exception as f:
        raise f
    if not success:
        raise e
