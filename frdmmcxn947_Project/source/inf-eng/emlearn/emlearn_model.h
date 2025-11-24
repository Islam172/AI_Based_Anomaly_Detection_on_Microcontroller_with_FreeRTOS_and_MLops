
#include <stdint.h>
#include "fsl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

status_t EMLEARN_MODEL_Init(void);

/**
 * inputData:  float* Fenster mit Rohdaten (Ax,Ay,Az) Länge CLSF_WINDOW*CLSF_CHANNELS
 *             Layout gemäß SENSOR_COLLECT_DATA_FORMAT:
 *             - INTERLEAVED : [Ax0,Ay0,Az0, Ax1,Ay1,Az1, ...]
 *             - BLOCKS     : [Ax0..AxN-1, Ay0..AyN-1, Az0..AzN-1]
 * size     :  Größe in Bytes von inputData
 * predClass:  Ausgabe-Klasse (0=normal, 1=anomalie)
 * tinf_us :  Inferenzzeit in µs (Feature+Modell)
 * verbose :  Debug-Ausgabe
 */
status_t EMLEARN_MODEL_RunInference(void *inputData, size_t size,
                                       int8_t *predClass, int32_t *tinf_us,
                                       uint8_t verbose);

#ifdef __cplusplus
}
#endif


