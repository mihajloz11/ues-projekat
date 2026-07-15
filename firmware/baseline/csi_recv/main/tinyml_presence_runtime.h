#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tinyml_presence_model.h"

// inferencija se radi na svakom četvrtom frejmu
#define TINYML_PRESENCE_INFER_EVERY_FRAMES 4

// dodaje jedan I/Q frejm i vraća true kada je spremna nova predikcija
bool tinyml_presence_runtime_push_iq(const int8_t *iq,
                                     size_t iq_len,
                                     float compensate_gain,
                                     tinyml_presence_result_t *result,
                                     int *latency_us,
                                     uint32_t *processed_frames);
