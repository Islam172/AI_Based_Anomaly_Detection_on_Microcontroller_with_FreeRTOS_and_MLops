#include "sensor_collect.h"
#include "sensor_raw.h"

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "timers.h"

#include "fsl_common.h"
#include "fsl_debug_console.h"
#include "fsl_gpio.h"
#include "board.h"


#include "app_config.h"




#if SENSOR_COLLECT_ACTION == SENSOR_COLLECT_RUN_INFERENCE
  #if SENSOR_COLLECT_RUN_INFENG == SENSOR_COLLECT_INFENG_TENSORFLOW
    #include "tfmodel.h"
    status_t (*SNS_MODEL_Init)(void) = &TFMODEL_Init;
    status_t (*SNS_MODEL_RunInference)(void *inputData, size_t size, int8_t *predClass, int32_t *tinf_us, uint8_t verbose) =
             &TFMODEL_RunInference;
  #elif SENSOR_COLLECT_RUN_INFENG == SENSOR_COLLECT_INFENG_EMLEARN
    #include "inf-eng/emlearn/emlearn_model.h"
    status_t (*SNS_MODEL_Init)(void) = &EMLEARN_MODEL_Init;
    status_t (*SNS_MODEL_RunInference)(void *inputData, size_t size, int8_t *predClass, int32_t *tinf_us, uint8_t verbose) =
             &EMLEARN_MODEL_RunInference;
  #endif               
  /* Input-Puffer für das Modell */
  static float g_clsfInputData[CLSF_CHANNELS * CLSF_WINDOW] __attribute__((aligned(32)));
#endif /* RUN_INFERENCE */


QueueHandle_t g_sensorCollectQueue = NULL;
TimerHandle_t g_sensorCollectTimer = NULL;

/* Einfache Sensordaten-Struktur für Queue */
typedef struct _sensor_data
{
    uint64_t sampleNum;
    uint64_t ts_us;
    int16_t rawDataSensor[3];  /* Ax, Ay, Az (roh, 16-bit) */
} sensor_data_t;

/* ---- Zeitbasis (optional; wird hier nur intern für ts_us verwendet) ---- */
uint64_t TIMER_GetTimeInUS(void)

{
    uint64_t us = ((SystemCoreClock / configTICK_RATE_HZ) - SysTick->VAL) / (SystemCoreClock / 1000000);
    us += (uint64_t)xTaskGetTickCount() * portTICK_PERIOD_MS * 1000;
    return us;
}

/* ---- CSV-Logger Task: Gibt NUR "Ax,Ay,Az\r\n" auf UART aus --------------- */
#if SENSOR_COLLECT_ACTION == SENSOR_COLLECT_LOG_EXT
static void SENSOR_Collect_LogExt_Task(void *pvParameters)
{
    sensor_data_t sensorData;

    /* Kein Header, keine Klasse, kein Timestamp: nur CSV Ax,Ay,Az */
    while (1)
    {
        if (g_sensorCollectQueue != NULL &&
            xQueueReceive(g_sensorCollectQueue, &sensorData, portMAX_DELAY) == pdPASS)
        {
            /* Rohwerte direkt als CSV ausgeben */
            PRINTF("%d,%d,%d\r\n",
                   sensorData.rawDataSensor[0],
                   sensorData.rawDataSensor[1],
                   sensorData.rawDataSensor[2]);
        }
    }

    vTaskDelete(NULL);
}
#endif /* LOG_EXT */

/* ---- Inference Task ------------------------------------------------------ */
#if SENSOR_COLLECT_ACTION == SENSOR_COLLECT_RUN_INFERENCE
static void SENSOR_Collect_RunInf_Task(void *pvParameters)
{
    status_t status = SNS_MODEL_Init();
    if (kStatus_Success != status)
    {
        PRINTF("[ERROR] Model init failed\r\n");
        vTaskDelete(NULL);
        return;
    }

    sensor_data_t sensorData;
    static uint16_t clsfSampIdx = 0;

    while (1)
    {
        if (g_sensorCollectQueue != NULL &&
            xQueueReceive(g_sensorCollectQueue, &sensorData, portMAX_DELAY) == pdPASS)
        {
            /* Ax, Ay, Az in g_clsfInputData kopieren — Format je nach Layout */

            for (int i = 0; i < CLSF_CHANNELS; i++)
            {
                float sens_val;

                /* WICHTIG:
                 * - TensorFlow-Pfad erwartet evtl. vor-normalisierte Rohdaten (3 Werte).
                 * - microMLgen-Pfad erwartet UNNORMIERTE Rohdaten (Fenster wird intern featurized + z-normalized).
                 */
            #if SENSOR_COLLECT_RUN_INFENG == SENSOR_COLLECT_INFENG_TENSORFLOW
                #if SENSOR_RAW_DATA_NORMALIZE
                    sens_val = (((float)sensorData.rawDataSensor[i]) - model_mean[i]) / model_std[i];
                #else
                    sens_val = (float)sensorData.rawDataSensor[i];

                #endif
            #elif SENSOR_COLLECT_RUN_INFENG == SENSOR_COLLECT_INFENG_EMLEARN
                /* Niemals hier normalisieren – micromlgen_engine macht Features + Z-Norm selbst */
                sens_val = (float)sensorData.rawDataSensor[i];
            #else
                sens_val = (float)sensorData.rawDataSensor[i];
            #endif

            #if SENSOR_COLLECT_DATA_FORMAT == SENSOR_COLLECT_DATA_FORMAT_INTERLEAVED
                g_clsfInputData[CLSF_CHANNELS * clsfSampIdx + i] = sens_val;
            #else /* BLOCKS */
                g_clsfInputData[i * CLSF_WINDOW + clsfSampIdx] = sens_val;
            #endif
            }

            if (++clsfSampIdx >= CLSF_WINDOW)
            {
                int32_t tinf_us = 0;
                int8_t  predClass = -1;

                SNS_MODEL_RunInference((void*)g_clsfInputData, sizeof(g_clsfInputData),
                                       &predClass, &tinf_us, SENSOR_COLLECT_INFENG_VERBOSE_EN);

                /* Ringpuffer: die letzten CLSF_OFFSET Samples behalten (HOP) */
            #if SENSOR_COLLECT_DATA_FORMAT == SENSOR_COLLECT_DATA_FORMAT_INTERLEAVED
                memcpy(&g_clsfInputData[0],
                       &g_clsfInputData[CLSF_CHANNELS * (CLSF_WINDOW - CLSF_OFFSET)],
                       CLSF_CHANNELS * CLSF_OFFSET * sizeof(g_clsfInputData[0]));
            #else /* BLOCKS */
                for (int i = 0; i < CLSF_CHANNELS; i++)
                {
                    memcpy(&g_clsfInputData[i * CLSF_WINDOW],
                           &g_clsfInputData[i * CLSF_WINDOW + (CLSF_WINDOW - CLSF_OFFSET)],
                           CLSF_OFFSET * sizeof(g_clsfInputData[0]));
                }
            #endif
                clsfSampIdx = CLSF_OFFSET;

                PRINTF("\rInference: t=%ld us, class=%d   ", tinf_us, predClass);

                /*if (predClass == 1) {

                	PRINTF("anomalie");

                } else {
                	PRINTF("normal");


                }*/


            }
        }
    }

    vTaskDelete(NULL);
}
#endif /* RUN_INFERENCE */


/* ---- Queue push helper --------------------------------------------------- */
static status_t SENSOR_Collect_PushData(const sensor_data_t *sensorData)
{
    if (g_sensorCollectQueue == NULL ||
        xQueueSend(g_sensorCollectQueue, (void*)sensorData, 0) != pdPASS)
    {
        /* Falls Producer schneller als Consumer: Meldung (optional) */
        // PRINTF("WARN Data Loss\r\n");
        return kStatus_Fail;
    }
    return kStatus_Success;
}

/* ---- Timer Callback: zieht Sample vom Sensor & in Queue ------------------ */
static void SENSOR_Collect_TimerCB(TimerHandle_t xTimer)
{
    static uint8_t first_run = 1;
    static uint64_t t0 = 0;
    static uint32_t num_dropped = 0;

    static sensor_data_t sensorData = { .sampleNum = 0 };

    if (first_run)
    {
        first_run = 0;
        t0 = TIMER_GetTimeInUS();
        return;
    }

    /* Sensor lesen: füllt int16[3] => Ax,Ay,Az */
    SENSOR_Run(&sensorData.rawDataSensor[0]);
    sensorData.sampleNum++;
    sensorData.ts_us = TIMER_GetTimeInUS() - t0;

    if (kStatus_Success != SENSOR_Collect_PushData(&sensorData))
    {
        /* Drop-Count nur als Debug nutzbar */
        num_dropped++;
        sensorData.sampleNum--;
    }
}

/* ---- MainTask: Init, Queue, Timer, Taskstart ----------------------------- */
void MainTask(void *pvParameters)

{


    status_t status = kStatus_Success;



    /* Define the init structure for the output LED pin*/
            gpio_pin_config_t led_config = {
                kGPIO_DigitalOutput,
                0,
            };


    /* Init output LED GPIO. */
            GPIO_PinInit(BOARD_LED_BLUE_GPIO, BOARD_LED_BLUE_GPIO_PIN, &led_config);
            /*GPIO_PinInit(BOARD_LED_RED_GPIO, BOARD_LED_RED_GPIO_PIN, &led_config);*/

    /*LEDs off */
    /*LED_BLUE_OFF();*/


    /* Sensor-HAL initialisieren (I2C, FXLS8974 verifizieren, ODR, FSR, …) */
    status = SENSOR_Init();
    if (status != kStatus_Success)
    {
        PRINTF("SENSOR_Init failed: %ld\r\n", status);
        goto main_task_exit;
    }

    /* Queue für Sensor-Daten */
    g_sensorCollectQueue = xQueueCreate(SENSOR_COLLECT_QUEUE_ITEMS, sizeof(sensor_data_t));
    if (g_sensorCollectQueue == NULL)
    {
        PRINTF("collect queue create failed!\r\n");
        status = kStatus_Fail;
        goto main_task_exit;
    }

    /* Arbeiter-Task je nach Modus */
#if SENSOR_COLLECT_ACTION == SENSOR_COLLECT_LOG_EXT
    if (xTaskCreate(SENSOR_Collect_LogExt_Task, "SENSOR_Collect_LogExt_Task",
                    4096, NULL, configMAX_PRIORITIES - 1, NULL) != pdPASS)
    {
        PRINTF("SENSOR_Collect_LogExt_Task creation failed!\r\n");
        status = kStatus_Fail;
        goto main_task_exit;
    }
#elif SENSOR_COLLECT_ACTION == SENSOR_COLLECT_RUN_INFERENCE
    if (xTaskCreate(SENSOR_Collect_RunInf_Task, "SENSOR_Collect_RunInf_Task",
                    4096, NULL, (tskIDLE_PRIORITY + 2), NULL) != pdPASS)
    {
        PRINTF("SENSOR_Collect_RunInf_Task creation failed!\r\n");
        status = kStatus_Fail;
        goto main_task_exit;
    }
#endif

    /* Periodischer Timer: treibt die Erfassung (Samplerate aus model_configuration.h) */
    g_sensorCollectTimer = xTimerCreate("g_sensorCollectTimer",
                                        (configTICK_RATE_HZ / SENSOR_COLLECT_RATE_HZ),
                                        pdTRUE, (void *)0, SENSOR_Collect_TimerCB);
    if (g_sensorCollectTimer == NULL)
    {
        PRINTF("collect timer create failed!\r\n");
        status = kStatus_Fail;
        goto main_task_exit;
    }
    if (xTimerStart(g_sensorCollectTimer, 0) != pdPASS)
    {
        PRINTF("collect timer start failed!\r\n");
        status = kStatus_Fail;
        goto main_task_exit;
    }

main_task_exit:
    PRINTF("MainTask exit, status %ld\r\n", status);
    vTaskDelete(NULL);
}
