@echo off
rem win/build.bat - configure and build NanoSeedLM with MSVC + CUDA + Ninja into build\ (Release).
rem   win\build.bat            build
rem   win\build.bat test       build and run the tests
rem Set NSLM_MODEL_DIR to a model folder before the first build to enable the tests that read one.
setlocal
rem (no paths inside parenthesised blocks: the ")" of "Program Files (x86)" would close them)
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined VSINSTALLDIR call :findvs || exit /b 1
cd /d "%~dp0.."
if not exist build\build.ninja call :configure || exit /b 1
cmake --build build || exit /b 1
if /i "%1"=="test" ctest --test-dir build --output-on-failure || exit /b 1
endlocal
exit /b 0

:configure
rem a label, not a parenthesised block: NSLM_MODEL_DIR may contain ")" (C:\Program Files (x86)\...)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DNSLM_MODEL_DIR=%NSLM_MODEL_DIR%" || exit /b 1
exit /b 0

:findvs
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\nslm_vsdir.txt"
set /p VSDIR=<"%TEMP%\nslm_vsdir.txt"
del "%TEMP%\nslm_vsdir.txt"
if not defined VSDIR (echo Visual Studio C++ build tools not found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
exit /b 0
