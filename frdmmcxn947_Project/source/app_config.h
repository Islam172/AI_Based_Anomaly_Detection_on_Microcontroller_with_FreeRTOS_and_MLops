
#ifndef _APP_CONFIG_H
#define _APP_CONFIG_H

/*******************************************************************************
 * Hardware configuration
 ******************************************************************************/
/*
 * DO NOT CHANGE!
 * AVAILABLE sensor boards
 */
#define ACCEL_4_CLICK_BOARD    1  //Uses FXLS8974 Accelerometer
#define FRDM_STBI_A8974_BOARD  2  //Uses FXLS8974 Accelerometer

/*
 * USER CONFIGURATION!
 * SELECT which sensor board is being with the FRDM-MCXN947
 */
#define SENSOR_BOARD           ACCEL_4_CLICK_BOARD

#if defined(SENSOR_BOARD) && (SENSOR_BOARD == FRDM_STBI_A8974_BOARD)
#elif defined(SENSOR_BOARD) && (SENSOR_BOARD == ACCEL_4_CLICK_BOARD)
#else
#error "ERROR: Undefined/unknown sensor board"
#endif /* SENSOR_BOARD */


/*******************************************************************************
 * Functional scenario configuration
 ******************************************************************************/
/*
 * DO NOT CHANGE!
 * AVAILABLE actions to be performed
 */
#define SENSOR_COLLECT_LOG_EXT                  1   // Collect and log data externally
#define SENSOR_COLLECT_RUN_INFERENCE            2   // Collect data and run inference

/*
 * DO NOT CHANGE!
 * AVAILABLE inference engine to be used
 */

#define SENSOR_COLLECT_INFENG_EMLEARN           1

/*
 * DO NOT CHANGE!
 * AVAILABLE data format to be used to feed the model
 */
#define SENSOR_COLLECT_DATA_FORMAT_BLOCKS       1   // Blocks of samples
#define SENSOR_COLLECT_DATA_FORMAT_INTERLEAVED  2   // Interleaved samples



/*
 * USER CONFIGURATION!
 * SELECT the action to be performed
 * Set to SENSOR_COLLECT_LOG_EXT to collect sensor data for training.
 * Set to SENSOR_COLLECT_RUN_INFERENCE to run ML inference.
 */
#define SENSOR_COLLECT_ACTION                   SENSOR_COLLECT_RUN_INFERENCE




#if SENSOR_COLLECT_ACTION == SENSOR_COLLECT_RUN_INFERENCE
/*
 * USER CONFIGURATION!
 * SELECT inference behavior
 */
//#define SENSOR_COLLECT_RUN_INFENG               SENSOR_COLLECT_INFENG_TENSORFLOW // Inference engine to be used
#define SENSOR_COLLECT_RUN_INFENG               SENSOR_COLLECT_INFENG_EMLEARN
#define SENSOR_RAW_DATA_NORMALIZE               0  // Normalize the raw data
#define SENSOR_COLLECT_INFENG_VERBOSE_EN        1   // Enable verbosity
#define SENSOR_COLLECT_DATA_FORMAT              SENSOR_COLLECT_DATA_FORMAT_INTERLEAVED



/*
 * DO NOT CHANGE!
 * Sanity check
 */
#if (SENSOR_COLLECT_RUN_INFENG != SENSOR_COLLECT_INFENG_EMLEARN)
#error "Unsupported inference engine"
#endif /* SENSOR_COLLECT_RUN_INFENG */
#else
#error "Unsupported action"
#endif /* SENSOR_COLLECT_ACTION */



#endif /* _APP_CONFIG_H */
