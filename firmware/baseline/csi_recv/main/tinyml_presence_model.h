#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TINYML_PRESENCE_WINDOW_SIZE 8
#define TINYML_PRESENCE_FRAME_AMPLITUDES 192
#define TINYML_PRESENCE_FEATURE_COUNT 391
#define TINYML_PRESENCE_HIDDEN_SIZE 32
#define TINYML_PRESENCE_CLASS_COUNT 2

// 1 = prije inferencije dijeli vektor obilježja globalnim prosjekom amplitude
// time se poništava drift apsolutnog nivoa; vrijednost mora odgovarati modelu i Python kodu
#define TINYML_PRESENCE_FEATURE_NORM_WINDOW_MEAN 1

typedef struct {
    const char *label;
    float empty_probability;
    float person_probability;
    float confidence;
    int predicted_index;
} tinyml_presence_result_t;

const char *tinyml_presence_model_source(void);
bool tinyml_presence_predict_from_feature(const float feature[TINYML_PRESENCE_FEATURE_COUNT],
                                          tinyml_presence_result_t *result);
