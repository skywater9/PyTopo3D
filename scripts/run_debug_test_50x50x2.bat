@echo off
cd /d "%~dp0.."

.venv\Scripts\python.exe main.py ^
  --nelx 50 ^
  --nely 50 ^
  --nelz 2 ^
  --volfrac 0.4 ^
  --penal 3.0 ^
  --rmin 3.0 ^
  --tolx 0.01 ^
  --maxloop 2000 ^
  --export-stl ^
  --force-field-preset debug_force_field_50x50x2 ^
  --material-preset pla_anisotropic ^
  --material-orientation-xyz xzy ^
  --support-mask-preset debug_support_mask_50x50x2 ^
  --protected-zones debug_protected_zone_50x50x2 ^
  --experiment-name debug_test_aniso_50x50x2 ^
  --description "Debug test with support on one end and force on the opposite end"
