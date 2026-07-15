#include "tinyml_presence_runtime.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "esp_timer.h"

// podrška za starije headere bez podešavanja normalizacije
#ifndef TINYML_PRESENCE_FEATURE_NORM_WINDOW_MEAN
#define TINYML_PRESENCE_FEATURE_NORM_WINDOW_MEAN 0
#endif

static float s_window[TINYML_PRESENCE_WINDOW_SIZE][TINYML_PRESENCE_FRAME_AMPLITUDES];
static float s_feature[TINYML_PRESENCE_FEATURE_COUNT];
static float s_scratch[TINYML_PRESENCE_WINDOW_SIZE * TINYML_PRESENCE_FRAME_AMPLITUDES];
static uint32_t s_processed_frames;
static size_t s_next_slot;
static size_t s_valid_frames;

static int compare_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) {
        return -1;
    }
    if (fa > fb) {
        return 1;
    }
    return 0;
}

// linearna interpolacija percentila iz već sortiranog niza
static float percentile_sorted(const float *sorted, size_t count, float percentile)
{
    if (count == 0) {
        return 0.0f;
    }
    if (count == 1) {
        return sorted[0];
    }

    float pos = (percentile / 100.0f) * (float)(count - 1);
    size_t lo = (size_t)floorf(pos);
    size_t hi = (size_t)ceilf(pos);
    float frac = pos - (float)lo;
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

// formira vektor obilježja: srednja vrijednost i standardna devijacija po podnosiocu
// zatim dodaje globalnu statistiku istim redoslijedom kao make_feature u Pythonu
static void build_feature(float feature[TINYML_PRESENCE_FEATURE_COUNT])
{
    const size_t total_count = TINYML_PRESENCE_WINDOW_SIZE * TINYML_PRESENCE_FRAME_AMPLITUDES;
    float global_sum = 0.0f;
    float global_sum_sq = 0.0f;
    float global_min = s_window[0][0];
    float global_max = s_window[0][0];
    size_t scratch_index = 0;

    for (size_t amp = 0; amp < TINYML_PRESENCE_FRAME_AMPLITUDES; amp++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (size_t frame = 0; frame < TINYML_PRESENCE_WINDOW_SIZE; frame++) {
            float value = s_window[frame][amp];
            sum += value;
            sum_sq += value * value;
            global_sum += value;
            global_sum_sq += value * value;
            if (value < global_min) {
                global_min = value;
            }
            if (value > global_max) {
                global_max = value;
            }
            s_scratch[scratch_index++] = value;
        }

        float mean = sum / (float)TINYML_PRESENCE_WINDOW_SIZE;
        float variance = sum_sq / (float)TINYML_PRESENCE_WINDOW_SIZE - mean * mean;
        feature[amp] = mean;
        feature[TINYML_PRESENCE_FRAME_AMPLITUDES + amp] = sqrtf(variance > 0.0f ? variance : 0.0f);
    }

    qsort(s_scratch, total_count, sizeof(float), compare_float);
    float global_mean = global_sum / (float)total_count;
    float global_variance = global_sum_sq / (float)total_count - global_mean * global_mean;
    size_t base = TINYML_PRESENCE_FRAME_AMPLITUDES * 2;
    feature[base + 0] = global_mean;
    feature[base + 1] = sqrtf(global_variance > 0.0f ? global_variance : 0.0f);
    feature[base + 2] = global_min;
    feature[base + 3] = global_max;
    feature[base + 4] = percentile_sorted(s_scratch, total_count, 50.0f);
    feature[base + 5] = percentile_sorted(s_scratch, total_count, 25.0f);
    feature[base + 6] = percentile_sorted(s_scratch, total_count, 75.0f);

#if TINYML_PRESENCE_FEATURE_NORM_WINDOW_MEAN
    // sva obilježja su linearna po amplitudi, pa dijeljenje globalnim prosjekom
    // odgovara normalizaciji prozora u make_windows()/make_feature(); global_mean postaje 1.0
    float norm_scale = feature[base + 0];
    if (norm_scale > 1e-6f) {
        for (size_t i = 0; i < TINYML_PRESENCE_FEATURE_COUNT; i++) {
            feature[i] /= norm_scale;
        }
    }
#endif
}

bool tinyml_presence_runtime_push_iq(const int8_t *iq,
                                     size_t iq_len,
                                     float compensate_gain,
                                     tinyml_presence_result_t *result,
                                     int *latency_us,
                                     uint32_t *processed_frames)
{
    if (iq == NULL || result == NULL || iq_len < TINYML_PRESENCE_FRAME_AMPLITUDES * 2) {
        return false;
    }

    int64_t start_us = esp_timer_get_time();
    // novi frejm ulazi u kružni bafer, a amplituda se računa iz I/Q para
    float *slot = s_window[s_next_slot];
    for (size_t amp = 0; amp < TINYML_PRESENCE_FRAME_AMPLITUDES; amp++) {
        float i_value = compensate_gain * (float)iq[amp * 2];
        float q_value = compensate_gain * (float)iq[amp * 2 + 1];
        slot[amp] = sqrtf(i_value * i_value + q_value * q_value);
    }

    s_next_slot = (s_next_slot + 1) % TINYML_PRESENCE_WINDOW_SIZE;
    if (s_valid_frames < TINYML_PRESENCE_WINDOW_SIZE) {
        s_valid_frames++;
    }
    s_processed_frames++;

    if (processed_frames != NULL) {
        *processed_frames = s_processed_frames;
    }
    // predikcija počinje tek kada se prozor napuni
    if (s_valid_frames < TINYML_PRESENCE_WINDOW_SIZE) {
        return false;
    }
    // računa se svaki N-ti frejm da bi se rasteretio procesor
    if ((s_processed_frames % TINYML_PRESENCE_INFER_EVERY_FRAMES) != 0) {
        return false;
    }

    build_feature(s_feature);
    bool ok = tinyml_presence_predict_from_feature(s_feature, result);
    if (latency_us != NULL) {
        *latency_us = (int)(esp_timer_get_time() - start_us);
    }
    return ok;
}
