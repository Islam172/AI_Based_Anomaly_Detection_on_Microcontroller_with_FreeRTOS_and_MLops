/*
 * accel_4_click_shield.h
 *
 *  Created on: Oct 16, 2025
 *      Author: islam_elmaaroufi
 */

#ifndef ACCEL_4_CLICK_SHIELD_H_
#define ACCEL_4_CLICK_SHIELD_H_

/* The shield name */
#define SHIELD_NAME "ACCEL_4_CLICK_SHIELD"
#define FXLS8974_I2C_ADDR   0x18

#define ACCEL4_SDA   P1_0
#define ACCEL4_SCL   P1_1

#include "fsl_lpi2c_cmsis.h"
extern ARM_DRIVER_I2C Driver_I2C3;
void LPI2C3_SignalEvent(uint32_t event);

#define I2C_S1_DRIVER        Driver_I2C3
#define I2C_S1_SignalEvent   LPI2C3_SignalEvent
#define I2C_S1_Device_Index  3
#define I2C_S1_SCL_PIN       ACCEL4_SCL
#define I2C_S1_SDA_PIN       ACCEL4_SDA

#endif /* ACCEL_4_CLICK_SHIELD_H_ */
