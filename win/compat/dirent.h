// win/compat/dirent.h - opendir / readdir / closedir on FindFirstFile (names only).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct dirent {
    char d_name[260];
};
typedef struct NslmDir DIR;

DIR* nslm_opendir(const char* path);
struct dirent* nslm_readdir(DIR* d);
int nslm_closedir(DIR* d);
#define opendir nslm_opendir
#define readdir nslm_readdir
#define closedir nslm_closedir

#ifdef __cplusplus
}
#endif
