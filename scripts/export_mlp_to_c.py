import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# simetrična int8 kvantizacija koristi najveću apsolutnu vrijednost podijeljenu sa 127
def quantize_symmetric(values):
    if values.size:
        max_abs = float(np.max(np.abs(values)))
    else:
        max_abs = 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return quantized, scale


# ispisuje float vrijednosti kao C niz sa više elemenata po liniji
def c_float_array(name, values, per_line=6):
    flat = values.astype(np.float32).reshape(-1)
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for start in range(0, len(flat), per_line):
        parts = []
        for v in flat[start : start + per_line]:
            parts.append(c_float_literal(float(v)))
        lines.append("    " + ", ".join(parts) + ",")
    lines.append("};")
    return "\n".join(lines)


# pretvara jedan float u C literal sa sufiksom f
def c_float_literal(value):
    text = f"{value:.8g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


# ispisuje int8 vrijednosti kao C niz
def c_int8_array(name, values, per_line=16):
    flat = values.astype(np.int8).reshape(-1)
    lines = [f"static const int8_t {name}[{len(flat)}] = {{"]
    for start in range(0, len(flat), per_line):
        parts = []
        for v in flat[start : start + per_line]:
            parts.append(str(int(v)))
        lines.append("    " + ", ".join(parts) + ",")
    lines.append("};")
    return "\n".join(lines)


# ispisuje imena klasa kao C niz stringova
def c_label_array(labels):
    parts = []
    for label in labels:
        safe = label.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'"{safe}"')
    values = ", ".join(parts)
    return f"static const char *const s_labels[{len(labels)}] = {{{values}}};"


# učitava JSON model i provjerava da je binarni model prisustva
def load_model(path):
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("model_type") != "amplitude_window_mlp":
        raise SystemExit(f"Expected amplitude_window_mlp, got {model.get('model_type')!r}")
    labels = model.get("labels")
    if labels != ["empty_room", "person_present"]:
        raise SystemExit(f"Expected binary presence labels, got {labels!r}")
    if model.get("activation") != "relu":
        raise SystemExit("C exporter podržava samo ReLU aktivaciju skrivenog sloja.")
    return model


def render_header(model, source_model):
    feature_count = int(model["feature_count"])
    window_size = int(model["window_size"])
    hidden_size = int(model["hidden_size"])
    frame_amplitudes = (feature_count - 7) // 2
    normalize_window = 1 if model.get("feature_norm") == "window_mean" else 0
    return f"""#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TINYML_PRESENCE_WINDOW_SIZE {window_size}
#define TINYML_PRESENCE_FRAME_AMPLITUDES {frame_amplitudes}
#define TINYML_PRESENCE_FEATURE_COUNT {feature_count}
#define TINYML_PRESENCE_HIDDEN_SIZE {hidden_size}
#define TINYML_PRESENCE_CLASS_COUNT 2

// 1 = prije inferencije dijeli vektor obilježja globalnim prosjekom amplitude
// time se poništava drift apsolutnog nivoa; vrijednost mora odgovarati modelu i Python kodu
#define TINYML_PRESENCE_FEATURE_NORM_WINDOW_MEAN {normalize_window}

typedef struct {{
    const char *label;
    float empty_probability;
    float person_probability;
    float confidence;
    int predicted_index;
}} tinyml_presence_result_t;

const char *tinyml_presence_model_source(void);
bool tinyml_presence_predict_from_feature(const float feature[TINYML_PRESENCE_FEATURE_COUNT],
                                          tinyml_presence_result_t *result);
"""


def render_source(model, source_model):
    labels = list(model["labels"])
    feature_count = int(model["feature_count"])
    hidden_size = int(model["hidden_size"])
    class_count = len(labels)

    scaler_mean = np.asarray(model["scaler"]["mean"], dtype=np.float32)
    scaler_std = np.asarray(model["scaler"]["std"], dtype=np.float32)
    scaler_std[scaler_std < 1e-6] = 1.0
    w1 = np.asarray(model["weights"]["w1"], dtype=np.float32)
    b1 = np.asarray(model["weights"]["b1"], dtype=np.float32)
    w2 = np.asarray(model["weights"]["w2"], dtype=np.float32)
    b2 = np.asarray(model["weights"]["b2"], dtype=np.float32)

    if w1.shape != (feature_count, hidden_size):
        raise SystemExit(f"Unexpected w1 shape {w1.shape}; expected {(feature_count, hidden_size)}")
    if w2.shape != (hidden_size, class_count):
        raise SystemExit(f"Unexpected w2 shape {w2.shape}; expected {(hidden_size, class_count)}")

    w1_q, w1_scale = quantize_symmetric(w1)
    w2_q, w2_scale = quantize_symmetric(w2)
    generated = utc_now()
    source_name = source_model.as_posix()

    return f"""#include "tinyml_presence_model.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

// generisano skriptom scripts/export_mlp_to_c.py, {generated}
// model: {source_name}
// težine su simetrični int8; scaler i bias ostaju float da se računanje obilježja
// na pločici poklapa sa Python kodom

#define W1_SCALE {w1_scale:.10g}f
#define W2_SCALE {w2_scale:.10g}f

{c_label_array(labels)}

{c_float_array("s_scaler_mean", scaler_mean)}

{c_float_array("s_scaler_std", scaler_std)}

{c_int8_array("s_w1_q", w1_q)}

{c_float_array("s_b1", b1)}

{c_int8_array("s_w2_q", w2_q)}

{c_float_array("s_b2", b2)}

const char *tinyml_presence_model_source(void)
{{
    return "{source_name}";
}}

// vraća stvarnu float težinu iz int8 tabele
static float dequant_w1(size_t feature_index, size_t hidden_index)
{{
    return (float)s_w1_q[feature_index * TINYML_PRESENCE_HIDDEN_SIZE + hidden_index] * W1_SCALE;
}}

static float dequant_w2(size_t hidden_index, size_t class_index)
{{
    return (float)s_w2_q[hidden_index * TINYML_PRESENCE_CLASS_COUNT + class_index] * W2_SCALE;
}}

bool tinyml_presence_predict_from_feature(const float feature[TINYML_PRESENCE_FEATURE_COUNT],
                                          tinyml_presence_result_t *result)
{{
    if (feature == NULL || result == NULL) {{
        return false;
    }}

    // skriveni sloj skalira ulaz, množi ga sa W1, dodaje bias i primjenjuje ReLU
    float hidden[TINYML_PRESENCE_HIDDEN_SIZE];
    for (size_t h = 0; h < TINYML_PRESENCE_HIDDEN_SIZE; h++) {{
        float acc = s_b1[h];
        for (size_t i = 0; i < TINYML_PRESENCE_FEATURE_COUNT; i++) {{
            float x = (feature[i] - s_scaler_mean[i]) / s_scaler_std[i];
            acc += x * dequant_w1(i, h);
        }}
        hidden[h] = acc > 0.0f ? acc : 0.0f;
    }}

    // izlazni sloj daje dva logita: prazna soba i prisutna osoba
    float logits[TINYML_PRESENCE_CLASS_COUNT];
    for (size_t c = 0; c < TINYML_PRESENCE_CLASS_COUNT; c++) {{
        float acc = s_b2[c];
        for (size_t h = 0; h < TINYML_PRESENCE_HIDDEN_SIZE; h++) {{
            acc += hidden[h] * dequant_w2(h, c);
        }}
        logits[c] = acc;
    }}

    // softmax pretvara dva logita u vjerovatnoće
    float max_logit = logits[0] > logits[1] ? logits[0] : logits[1];
    float exp0 = expf(logits[0] - max_logit);
    float exp1 = expf(logits[1] - max_logit);
    float denom = exp0 + exp1;
    if (denom <= 0.0f || !isfinite(denom)) {{
        return false;
    }}

    float empty_probability = exp0 / denom;
    float person_probability = exp1 / denom;
    int predicted_index = person_probability >= empty_probability ? 1 : 0;

    result->label = s_labels[predicted_index];
    result->empty_probability = empty_probability;
    result->person_probability = person_probability;
    result->confidence = predicted_index == 1 ? person_probability : empty_probability;
    result->predicted_index = predicted_index;
    return true;
}}
"""


def main():
    parser = argparse.ArgumentParser(description="Export a binary presence MLP to ESP32-S3 C/int8 files.")
    parser.add_argument("--model", default="models/presence_csi_mlp_fast.json")
    parser.add_argument("--out-dir", default="firmware/baseline/csi_recv/main")
    parser.add_argument("--basename", default="tinyml_presence_model")
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    header_path = out_dir / f"{args.basename}.h"
    source_path = out_dir / f"{args.basename}.c"
    header_path.write_text(render_header(model, model_path), encoding="utf-8")
    source_path.write_text(render_source(model, model_path), encoding="utf-8")

    print(
        json.dumps(
            {
                "model": str(model_path),
                "header": str(header_path),
                "source": str(source_path),
                "window_size": model["window_size"],
                "feature_count": model["feature_count"],
                "hidden_size": model["hidden_size"],
                "labels": model["labels"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
