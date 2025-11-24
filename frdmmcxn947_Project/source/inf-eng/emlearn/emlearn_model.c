// emlearn_model.c — RandomForest (emlearn) mit p(normal) + Schwelle
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "fsl_common.h"
#include "fsl_debug_console.h"
#include "sensor_collect.h"
#include "emlearn_model.h"

// ---- Projekt-Header (Preprocessing & Layout) ----
#include "app_config.h"                     // CLSF_CHANNELS, CLSF_WINDOW, CLSF_OFFSET, SENSOR_COLLECT_DATA_FORMAT_*
#include "inf-eng/emlearn/axis_scaler.h"    // AXIS_MEAN[3], AXIS_STD[3]

// ---- Dein generiertes emlearn-RF-Modell ----
// Der Generator hat Funktionen rf_model_predict(...) und rf_model_predict_proba(...)
// Stelle sicher, dass dieser Header im Include-Pfad liegt.
#include "models/emlearn/model.h"           // deklariert rf_model_predict_proba(...)

// ---- Schwelle wie in Python (p(normal) >= thr -> NORMAL) ----
#ifndef RF_THRESHOLD
#define RF_THRESHOLD 0.50f
#endif

// -------- helpers --------
static inline void dwt_init(void){
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_cycles(void){ return DWT->CYCCNT; }

static inline float mean_f(const float *x, int n){
    double s=0.0; for (int i=0; i<n; i++) s += x[i];
    return (float)(s / (double)n);
}
static inline float std_f(const float *x, int n){
    float mu = mean_f(x,n);
    double s=0.0; for (int i=0; i<n; i++){ double d = (double)x[i]-mu; s += d*d; }
    return (float)sqrt(s / (double)n); // ddof=0 — wie im Notebook
}

// Ein Sample (Ax,Ay,Az) aus Eingabepuffer holen (Layout!) + Z-Score pro Achse
static inline void fetch_norm_sample(const float *in, int s, float out3[3]){
#if SENSOR_COLLECT_DATA_FORMAT == SENSOR_COLLECT_DATA_FORMAT_INTERLEAVED
    const float ax = in[CLSF_CHANNELS*s + 0];
    const float ay = in[CLSF_CHANNELS*s + 1];
    const float az = in[CLSF_CHANNELS*s + 2];
#else // BLOCKS
    const float ax = in[0*CLSF_WINDOW + s];
    const float ay = in[1*CLSF_WINDOW + s];
    const float az = in[2*CLSF_WINDOW + s];
#endif
    out3[0] = (ax - AXIS_MEAN[0]) / AXIS_STD[0];
    out3[1] = (ay - AXIS_MEAN[1]) / AXIS_STD[1];
    out3[2] = (az - AXIS_MEAN[2]) / AXIS_STD[2];
}

// -------- API --------
status_t EMLEARN_MODEL_Init(void){
    dwt_init();
    return kStatus_Success;
}

status_t EMLEARN_MODEL_RunInference(void *inputData, size_t size,
                                    int8_t *predClass, int32_t *tinf_us,
                                    uint8_t verbose)
{
    if (!inputData || !predClass || !tinf_us) return kStatus_Fail;

    const size_t need_bytes = (size_t)CLSF_CHANNELS * (size_t)CLSF_WINDOW * sizeof(float);
    if (size < need_bytes){
        if (verbose){
            PRINTF("[emlearn/RF] bad input size: have %u, need %u\r\n",
                   (unsigned)size, (unsigned)need_bytes);
        }
        return kStatus_Fail;
    }

    const float *in = (const float*)inputData;

    static float ax_buf[CLSF_WINDOW];
    static float ay_buf[CLSF_WINDOW];
    static float az_buf[CLSF_WINDOW];

    uint32_t c0 = dwt_cycles();

    // 1) Rohdaten → Z-Score pro Achse → Fensterpuffer
    for (int s=0; s<CLSF_WINDOW; s++){
        float v[3];
        fetch_norm_sample(in, s, v);
        ax_buf[s]=v[0]; ay_buf[s]=v[1]; az_buf[s]=v[2];
    }

    // 2) Features (mean & std je Achse) → 6D
    float feat[6];
    feat[0]=mean_f(ax_buf,CLSF_WINDOW); feat[1]=std_f(ax_buf,CLSF_WINDOW);
    feat[2]=mean_f(ay_buf,CLSF_WINDOW); feat[3]=std_f(ay_buf,CLSF_WINDOW);
    feat[4]=mean_f(az_buf,CLSF_WINDOW); feat[5]=std_f(az_buf,CLSF_WINDOW);

    // 3) RF-Probabilitäten holen: out[0]=p(anomalie), out[1]=p(normal)
    float probs[2] = {0.0f, 0.0f};
    int err = rf_model_predict_proba(feat, 6, probs, 2);
    if (err != 0){
        // Fallback: nur Klasse
        int cls = rf_model_predict(feat, 6);
        *predClass = (int8_t)cls;
        uint32_t c1_fb = dwt_cycles();
        *tinf_us = (int32_t)((float)(c1_fb - c0) * 1e6f / (float)SystemCoreClock);
        if (verbose){
            PRINTF("[emlearn/RF] (no proba) cls=%d  feat: %.4f %.4f | %.4f %.4f | %.4f %.4f  (%d us)\r\n",
                   *predClass, feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], (int)*tinf_us);
        }
        return kStatus_Success;
    }

    const float p_normal = probs[1]; // class_order = [0,1] => 1 = normal
    *predClass = (p_normal >= RF_THRESHOLD) ? 1 : 0;

    uint32_t c1 = dwt_cycles();
    *tinf_us = (int32_t)((float)(c1 - c0) * 1e6f / (float)SystemCoreClock);

    if (verbose) {
        PRINTF("[emlearn/RF] p(normal)=%.3f thr=%.2f => %s  (%d us)\r\n",
               (double)p_normal, (double)RF_THRESHOLD,
               *predClass ? "normal" : "ANOMALIE", (int)*tinf_us);
        PRINTF("[emlearn/RF] feat: %.4f %.4f | %.4f %.4f | %.4f %.4f\r\n",
               feat[0], feat[1], feat[2], feat[3], feat[4], feat[5]);
    }

    return kStatus_Success;
}
