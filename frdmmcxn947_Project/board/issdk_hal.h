
/**
 * @file issdk_hal.h
 * @brief Wrapper for Hardware Abstraction Layer (HAL)

    This file simply provides one level of indirection for the developer
    to select the particular Hardware Abstraction Layer they would like to use.
*/

#ifndef __ISSDK_HAL_H__
#define __ISSDK_HAL_H__

#include "fsl_lpi2c_cmsis.h"
#include "fsl_lpspi_cmsis.h"
#include "fsl_lpuart_cmsis.h"

#include "frdmmcxn947.h"            //Include appropriate MCU board header file
#include "accel_4_click_shield.h" //Include appropriate sensor shield board header file


#define I2C_S_SCL_PIN      I2C_S1_SCL_PIN
#define I2C_S_SDA_PIN      I2C_S1_SDA_PIN
#define I2C_S_DRIVER       I2C_S1_DRIVER
#define I2C_S_SIGNAL_EVENT I2C_S1_SIGNAL_EVENT
#define I2C_S_DEVICE_INDEX I2C_S1_DEVICE_INDEX

#endif // __ISSDK_HAL_H__
