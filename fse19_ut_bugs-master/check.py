import time, sys, os
from __main__ import start_time
def my_excepthook(type, value, traceback):
    end = time.time()
    print("Program crashed after", end - start_time, "seconds")
    sys.__excepthook__(type, value, traceback)  # Print error message

sys.excepthook = my_excepthook