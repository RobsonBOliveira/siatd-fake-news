from pathlib import Path
import shutil

for folder in ["models", "output"]:
    shutil.rmtree(folder, ignore_errors=True)

for p in Path(".").rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

for p in Path(".").rglob("*.pyc"):
    p.unlink(missing_ok=True)

print("Limpeza concluída!")