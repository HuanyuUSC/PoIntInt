import sys
import os

# Add the build directory to the Python path
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cpp', 'build', 'Release'))
sys.path.append(build_path)

try:
    import minimal
    print("Successfully imported 'minimal' module.")
    message = minimal.say_hello()
    print(f"minimal.say_hello() returned: '{message}'")
    assert message == "Hello, World!"
    print("Test PASSED!")
except ImportError as e:
    print(f"Error: Could not import the 'minimal' module.")
    print(f"ImportError: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()
