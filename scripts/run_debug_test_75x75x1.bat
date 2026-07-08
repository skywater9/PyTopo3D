@echo off
cd /d "%~dp0.."

.venv\Scripts\python.exe main.py ^
  --nelx 75 ^
  --nely 75 ^
  --nelz 1 ^
  --volfrac 0.4 ^
  --penal 3.0 ^
  --rmin 3.0 ^
  --tolx 0.01 ^
  --maxloop 2000 ^
  --export-stl ^
  --export-mode blocky ^
  --force-field-preset debug_force_field_75x75x1 ^
  --material-preset pla_xanisotropic ^
  --material-orientation-xyz xzy ^
  --support-mask-preset debug_support_mask_75x75x1 ^
  --protected-zones debug_protected_zone_75x75x1 ^
  --eval-material-presets pla_isotropic pla_anisotropic pla_xanisotropic ^
  --experiment-name debug_test_xaniso_75x75x1 ^
  --description "Debug test with support on one end and force on the opposite end"
