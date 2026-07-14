@echo off
cd /d "%~dp0.."

.venv\Scripts\python.exe main.py ^
  --nelx 30 ^
  --nely 140 ^
  --nelz 4 ^
  --volfrac 0.5 ^
  --penal 3.0 ^
  --rmin 3.0 ^
  --tolx 0.01 ^
  --maxloop 200 ^
  --elem-size 0.001 ^
  --export-stl ^
  --create-animation ^
  --force-field-preset tensile_test ^
  --material-preset pla_anisotropic ^
  --support-mask-preset tensile_test ^
  --protected-zones tensile_support_face tensile_load_face ^
  --experiment-name tensile_test_aniso_30x140x4 ^
  --description "Tensile test with support on one end and force on the opposite end"
