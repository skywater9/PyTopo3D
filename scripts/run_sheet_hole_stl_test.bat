@echo off
cd /d "%~dp0.."

.venv\Scripts\python.exe main.py ^
	--design-space-stl "results\prepared_stl\sheet_hole_aligned_for_4x50x160.stl" ^
	--target-nelx 4 ^
	--nelx 4 ^
	--nely 50 ^
	--nelz 160 ^
    --elem-size 0.0005 ^
	--skip-optimization ^
	--export-stl ^
	--material-preset pla_anisotropic ^
	--force-field-preset sheet_hole_4x50x160 ^
	--support-mask-preset sheet_hole_4x50x160 ^
    --eval-material-presets pla_isotropic pla_anisotropic pla_xanisotropic ^
	--experiment-name sheet_hole_stl_test_4x50x160 ^
	--description "Custom STL sheet-hole test, no optimization, support/load on opposite sides"
