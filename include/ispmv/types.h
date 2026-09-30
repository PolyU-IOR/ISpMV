#ifndef ISPMV_TYPES_H
#define ISPMV_TYPES_H
#include <stdint.h>
#ifdef _WIN32
# ifdef ISPMV_BUILD
#  define ISPMV_API __declspec(dllexport)
# else
#  define ISPMV_API __declspec(dllimport)
# endif
#else
# define ISPMV_API __attribute__((visibility("default")))
#endif
typedef struct ispmv_plan ispmv_plan;
typedef struct ispmv_device ispmv_device;
typedef enum ispmv_status {
    ISPMV_SUCCESS = 0,
    ISPMV_INVALID_ARGUMENT = 1,
    ISPMV_OUT_OF_MEMORY = 2,
    ISPMV_CUDA_ERROR = 3,
    ISPMV_CORE_ERROR = 4,
    ISPMV_UNSUPPORTED = 5
} ispmv_status;
typedef enum ispmv_index_base {
    ISPMV_INDEX_ZERO = 0,
    ISPMV_INDEX_ONE = 1
} ispmv_index_base;
#endif
