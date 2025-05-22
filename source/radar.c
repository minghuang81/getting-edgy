

#include <inttypes.h>
#include <stdio.h>
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
/* FreeRTOS header file */
#include <FreeRTOS.h>
#include <task.h>

//#include "xensiv_bgt60trxx_mtb.h"
//#define XENSIV_BGT60TRXX_CONF_IMPL
#include "resource_map.h"
#include "xensiv_bgt60trxx_mtb.h"
#include <doppler30.h>

/*******************************************************************************
* Macros
*******************************************************************************/

/* enable 1V8 LDO on radar wingboard*/
#define PIN_XENSIV_BGT60TRXX_LDO_EN         CYBSP_GPIO5
#define XENSIV_BGT60TRXX_SPI_FREQUENCY      (25000000UL)
#define NUM_SAMPLES_PER_FRAME               (XENSIV_BGT60TRXX_CONF_NUM_RX_ANTENNAS *\
                                             XENSIV_BGT60TRXX_CONF_NUM_CHIRPS_PER_FRAME *\
                                             XENSIV_BGT60TRXX_CONF_NUM_SAMPLES_PER_CHIRP)
#define FRAME_SEG							(512)	/* 1024 bytes to fill and fit in one UDP */
#define FRAME_DIV							(NUM_SAMPLES_PER_FRAME / FRAME_SEG)

/*******************************************************************************
* Global variables
*******************************************************************************/
/* UDP server task handle. */
extern TaskHandle_t server_task_handle;
/* RADAR task handle. */
extern TaskHandle_t radar_task_handle;

/* buffer of radar data to send over UDP socket, filled by the notifying process */
extern uint8 *pBufToSend;
extern uint32_t szBufToSend;
/* command to send over UDP socket, filled by the notifying process */
extern uint8 *pCmdToSend;
extern uint32_t szCmdToSend;
extern const uint32_t typeToSend_cmd;
extern const uint32_t typeToSend_data;
extern bool client_connected;

extern void print_heap_usage(char *msg);

static cyhal_spi_t cyhal_spi;
static xensiv_bgt60trxx_mtb_t sensor;
/* Allocate enough memory for the radar dara frame. */
static uint16_t samples[NUM_SAMPLES_PER_FRAME];


/* Interrupt handler to react on sensor indicating the availability of new data */
#if defined(CYHAL_API_VERSION) && (CYHAL_API_VERSION >= 2)
void xensiv_bgt60trxx_mtb_interrupt_handler(void *args, cyhal_gpio_event_t event)
#else
void xensiv_bgt60trxx_mtb_interrupt_handler(void *args, cyhal_gpio_irq_event_t event)
#endif
{
    CY_UNUSED_PARAMETER(args);
    CY_UNUSED_PARAMETER(event);
    bool data_available = true;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xTaskNotifyFromISR(radar_task_handle, data_available,
    		eSetValueWithoutOverwrite, &xHigherPriorityTaskWoken);
    /* Force a context switch if xHigherPriorityTaskWoken is now set to pdTRUE. */
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void radar_task(void *arg)
{
	cy_rslt_t result = CY_RSLT_SUCCESS;
    printf("XENSIV BGT60TRxx Starting ...\r\n");

    /* Initialize the SPI interface to BGT60. */
    result = cyhal_spi_init(&cyhal_spi,
                            PIN_XENSIV_BGT60TRXX_SPI_MOSI,
                            PIN_XENSIV_BGT60TRXX_SPI_MISO,
                            PIN_XENSIV_BGT60TRXX_SPI_SCLK,
                            NC,
                            NULL,
                            8,
                            CYHAL_SPI_MODE_00_MSB,
                            false);
    CY_ASSERT(result == CY_RSLT_SUCCESS);
    /* Reduce drive strength to improve EMI */
    Cy_GPIO_SetSlewRate(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_MOSI),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_MOSI), CY_GPIO_SLEW_FAST);
    Cy_GPIO_SetDriveSel(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_MOSI),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_MOSI), CY_GPIO_DRIVE_1_8);
    Cy_GPIO_SetSlewRate(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_SCLK),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_SCLK), CY_GPIO_SLEW_FAST);
    Cy_GPIO_SetDriveSel(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_SCLK),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_SCLK), CY_GPIO_DRIVE_1_8);
    /* Set SPI data rate to communicate with sensor */
    result = cyhal_spi_set_frequency(&cyhal_spi, XENSIV_BGT60TRXX_SPI_FREQUENCY);
    CY_ASSERT(result == CY_RSLT_SUCCESS);
    /* Enable the LDO. */
//    result = cyhal_gpio_init(PIN_XENSIV_BGT60TRXX_LDO_EN,
//                             CYHAL_GPIO_DIR_OUTPUT,
//                             CYHAL_GPIO_DRIVE_STRONG,
//                             true);
//    CY_ASSERT(result == CY_RSLT_SUCCESS);
    /* Wait LDO stable */
    (void)cyhal_system_delay_ms(5);

    result = xensiv_bgt60trxx_mtb_init(&sensor,
                                       &cyhal_spi,
                                       PIN_XENSIV_BGT60TRXX_SPI_CSN,
                                       PIN_XENSIV_BGT60TRXX_RSTN,
                                       register_list,
                                       XENSIV_BGT60TRXX_CONF_NUM_REGS);
    CY_ASSERT(result == CY_RSLT_SUCCESS);
    /* The sensor will generate an interrupt once the sensor FIFO level is
       NUM_SAMPLES_PER_FRAME */
    result = xensiv_bgt60trxx_mtb_interrupt_init(&sensor,
                                                 FRAME_SEG,
                                                 PIN_XENSIV_BGT60TRXX_IRQ,
                                                 CYHAL_ISR_PRIORITY_DEFAULT,
                                                 xensiv_bgt60trxx_mtb_interrupt_handler,
                                                 NULL);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    if (xensiv_bgt60trxx_start_frame(&sensor.dev, true) != XENSIV_BGT60TRXX_STATUS_OK)
    {
        CY_ASSERT(0);
    }
    uint32_t frame_idx = 0;
    for(;;)
    {
    	bool sendToClient = client_connected;	// flag valid for one frame only
    	for (int k=0; k<FRAME_DIV; k++) {
			/* Wait for the radar device to indicate the availability of the data to fetch. */
    		uint32_t type;
    		xTaskNotifyWait(0, 0, &type, portMAX_DELAY);
			if (xensiv_bgt60trxx_get_fifo_data(&sensor.dev, &samples[k*FRAME_SEG],
                                               FRAME_SEG) != XENSIV_BGT60TRXX_STATUS_OK)
			{
				printf("Error xensiv_bgt60trxx_get_fifo_data\r\n");
				CY_ASSERT(0);
			}
			// Broadcast to connected client
			if (sendToClient) {
				pBufToSend = (uint8 *)&samples[k*FRAME_SEG];
				szBufToSend = FRAME_SEG*2;
				xTaskNotify(server_task_handle, typeToSend_data, eSetValueWithoutOverwrite);
			}
    	}
        frame_idx++;
        char msg[64];
        sprintf(msg, "Captured frame#%lu\n", frame_idx);
        printf(msg);
        //print_heap_usage(msg);
    }
}
