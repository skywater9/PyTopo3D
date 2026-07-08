@echo off
cd /d "%~dp0.."

.venv\Scripts\python.exe main.py ^
  --nelx 20 ^
  --nely 20 ^
  --nelz 100 ^
  --skip-optimization ^
  --elem-size 0.001 ^
  --export-stl ^
  --export-mode blocky ^
  --force-field-preset debug_force_field_20x20x100 ^
  --material-preset pla_anisotropic ^
  --support-mask-preset debug_support_mask_20x20x100 ^
  --eval-material-presets pla_isotropic pla_anisotropic pla_xanisotropic ^
  --experiment-name debug_test_fill_aniso_20x20x100 ^
  --description "Debug test with support on one end and force on the opposite end"
