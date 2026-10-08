@echo off
rem win/build.bat - configure and build NanoSeedLM with MSVC + CUDA + Ninja into build\ (Release).
rem   win\build.bat            build
rem   win\build.bat test       build and run the tests
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo Visual Studio C++ build tools not found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0.."
if not exist build\build.ninja (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DNSLM_MODEL_DIR=%NSLM_MODEL_DIR%" || exit /b 1
)
cmake --build build || exit /b 1
if /i "%1"=="test" ctest --test-dir build --output-on-failure || exit /b 1
endlocal
