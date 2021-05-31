# import subprocess as sb
# import os

# ord_dir = os.getcwd()
# os.chdir("SIFTflow")
# print(os.getcwd())

# same_command = ["python", "kong_use_evalUnwarp.py"]
# sb.run(same_command)

# os.chdir(ord_dir)
# print(os.getcwd())


import os
import sys
sys.path.append("SIFTflow")
from kong_use_evalUnwarp import use_DewarpNet_eval

ord_dir = os.getcwd()
os.chdir("SIFTflow")
print(os.getcwd())

use_DewarpNet_eval(path1="", path2="")

os.chdir(ord_dir)
print(os.getcwd())
