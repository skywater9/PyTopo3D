@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe main.py ^
  --nelx 20 ^
  --nely 20 ^
  --nelz 60 ^
  --volfrac 0.2 ^
  --penal 3.0 ^
  --rmin 3.0 ^
  --tolx 0.05 ^
  --maxloop 2000 ^
  --export-stl ^
  --force-field-preset cantilever_test ^
  --material-preset pla_isotropic ^
  --support-mask-preset cantilever_test ^
  --protected-zones cantilever_support_face cantilever_load_face ^
  --experiment-name cantilever_test_iso_20x20x60 ^
  --description "Cantilever test with support on one end and force on the opposite end"
