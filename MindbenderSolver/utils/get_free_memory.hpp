#pragma once


/// https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g
static unsigned long long getTotalSystemMemory();

static bool hasAtLeastGBMemoryTotal(unsigned int theGB);


#ifdef WIN32
#include <windows.h>
unsigned long long getTotalSystemMemory() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
}
#elif UNIX
#include <unistd.h>

unsigned long long getTotalSystemMemory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif

bool hasAtLeastGBMemoryTotal(const unsigned int theGB) {
    const unsigned long long mem = getTotalSystemMemory();
    const unsigned long long memGB = mem / 1024 / 1024 / 1024;
    return theGB <= memGB - 1;
}


