'''
Sets up imports so test folder can import from src folder
'''
import sys
import os
try:
    from _label_ids import _CURE_LABEL  # type: ignore
except ImportError:
    __CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    __PARENT_DIR = os.path.dirname(__CURRENT_DIR)
    __SRC_DIR = os.path.join(__PARENT_DIR, "src")
    sys.path.append(__SRC_DIR)
