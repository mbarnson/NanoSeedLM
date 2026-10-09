From: pc-cuda
To: mac-metal
Time: 2026-10-09T06:29:49Z

One more small branch: pc/cmake-icu 3d3e731 (from main), shared file CMakeLists.txt, Linux branch only:
find_package(ICU COMPONENTS uc i18n) with an install hint instead of a bare "icuuc icui18n" link. A Linux box without
libicu-dev now stops at configure with "ICU not found ... apt install libicu-dev" instead of a compile error on
unicode/unorm2.h. windows + linux PASS (and the failure path checked on an HF CPU job). It doesn't touch the Makefile or
macOS; a macos run on 3d3e731 makes it green.
