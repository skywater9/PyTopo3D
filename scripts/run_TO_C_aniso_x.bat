@echo off
cd /d "%~dp0.."

.venv\Scripts\python.exe main.py ^
  --nelx 180 ^
  --nely 130 ^
  --nelz 1 ^
  --x-axis-scale 0.18
  --y-axis-scale 0.13
  --z-axis-scale 0.004 ^
  --volfrac 0.4 ^
  --penal 3.0 ^
  --rmin 3.0 ^
  --tolx 0.01 ^
  --maxloop 200 ^
  --export-stl ^
  --material-preset pla_aniso_experimental ^
  --material-orientation-xyz xyz ^
  --force-field-preset force_field_TO_C ^
  --support-mask-preset support_mask_TO_C ^
  --obstacle-ranges obstacle_range_TO_C_1 ^
  --experiment-name TO_C_aniso_x ^
