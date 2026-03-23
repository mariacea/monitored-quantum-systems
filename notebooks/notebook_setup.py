import sys
import os

# Añade la raíz del proyecto al sys.path automáticamente
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
os.chdir("..")