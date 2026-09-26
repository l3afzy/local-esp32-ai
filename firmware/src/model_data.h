// The model file, embedded in flash by embed_model.py (see platformio.ini).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
extern const uint8_t model_data[];
extern const uint8_t model_data_end[];
#ifdef __cplusplus
}
#endif

#define model_data_len ((unsigned)(model_data_end - model_data))
