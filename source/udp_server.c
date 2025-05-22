/******************************************************************************
* File Name:   udp_server.c
*
* Description: This file contains declaration of task and functions related to
*              UDP server operation.
*
********************************************************************************
* Copyright 2020-2024, Cypress Semiconductor Corporation (an Infineon company) or
* an affiliate of Cypress Semiconductor Corporation.  All rights reserved.
*
* This software, including source code, documentation and related
* materials ("Software") is owned by Cypress Semiconductor Corporation
* or one of its affiliates ("Cypress") and is protected by and subject to
* worldwide patent protection (United States and foreign),
* United States copyright laws and international treaty provisions.
* Therefore, you may use this Software only as provided in the license
* agreement accompanying the software package from which you
* obtained this Software ("EULA").
* If no EULA applies, Cypress hereby grants you a personal, non-exclusive,
* non-transferable license to copy, modify, and compile the Software
* source code solely for use in connection with Cypress's
* integrated circuit products.  Any reproduction, modification, translation,
* compilation, or representation of this Software except as specified
* above is prohibited without the express written permission of Cypress.
*
* Disclaimer: THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, NONINFRINGEMENT, IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. Cypress
* reserves the right to make changes to the Software without notice. Cypress
* does not assume any liability arising out of the application or use of the
* Software or any product or circuit described in the Software. Cypress does
* not authorize its products for use in any products where a malfunction or
* failure of the Cypress product may reasonably be expected to result in
* significant property damage, injury or death ("High Risk Product"). By
* including Cypress's product in a High Risk Product, the manufacturer
* of such system or application assumes all risk of such use and in doing
* so agrees to indemnify Cypress against all liability.
*******************************************************************************/

/* Header file includes */
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <inttypes.h>

/* FreeRTOS header file */
#include <FreeRTOS.h>
#include <task.h>

/* Cypress secure socket header file */
#include "cy_secure_sockets.h"

/* Wi-Fi connection manager header files */
#include "cy_wcm.h"
#include "cy_wcm_error.h"

/* UDP server task header file. */
#include "udp_server.h"

/*******************************************************************************
* Macros
********************************************************************************/
/* RTOS related macros for UDP server task. */
#define RTOS_TASK_TICKS_TO_WAIT                   (1000)

/* Length of the LED ON/OFF command issued from the UDP server. */
#define UDP_LED_CMD_LEN                           (1)

/* LED ON and LED OFF commands. */
#define LED_ON_CMD                                '1'
#define LED_OFF_CMD                               '0'

#define LED_ON_ACK_MSG                            "LED ON ACK"

/* Initial message sent to UDP Server to confirm client availability. */
#define START_FRAME_MSG                           "A"
/* End message sent to UDP Server to stop broadcast by the server. */
#define END_FRAME_MSG                             "Z"
/* Client message sent to UDP Server to request for the server's addr */
#define DISCOVERY_MSG							  "DISCOVERY_REQUEST"

/* Interrupt priority of the user button. */
#define USER_BTN_INTR_PRIORITY                    (5)

/* Buffer size to store the incoming messages from server, in bytes. */
#define MAX_UDP_RECV_BUFFER_SIZE                  (20)

/*******************************************************************************
* Function Prototypes
********************************************************************************/
static cy_rslt_t connect_to_wifi_ap(void);
static cy_rslt_t create_udp_server_socket(void);
static cy_rslt_t udp_server_recv_handler(cy_socket_t socket_handle, void *arg);
static void isr_button_press( void *callback_arg, cyhal_gpio_event_t event);
void print_heap_usage(char *msg);

/*******************************************************************************
* Global Variables
********************************************************************************/
/* Secure socket variables. */
cy_socket_t server_handle;
cy_socket_sockaddr_t udp_server_addr, peer_addr, peer_addr_cmd;
char server_addr_s[16];
char *addrToStr(cy_socket_sockaddr_t *addr) {
	static char b[16];
	uint32 v = addr->ip_address.ip.v4;
	sprintf(b,"%d.%d.%d.%d", (uint8)v, (uint8)(v >> 8), (uint8)(v >> 16),(uint8)(v >> 24));
	return b;
}
char *peer_addr_s()     { return addrToStr(&peer_addr); }
char *peer_addr_cmd_s() { return addrToStr(&peer_addr_cmd); }
/* Flag variable to track client connection status,
 * set to True when START_FRAME_MSG is received from client. */
bool client_connected = false;

/* Flags to tack the LED state and command. */
bool led_state = CYBSP_LED_STATE_OFF;

/* UDP Server task handle. */
extern TaskHandle_t server_task_handle;

cyhal_gpio_callback_data_t cb_data =
{
.callback = isr_button_press,
.callback_arg = NULL
};

/* buffer of radar data to send over UDP socket, filled by the notifying process */
uint8 *pBufToSend;
uint32_t szBufToSend = 0;
/* c0mmand to send over UDP socket, filled by the notifying process */
uint8 *pCmdToSend;
uint32_t szCmdToSend = 0;
const uint32_t typeToSend_cmd = 0;
const uint32_t typeToSend_data = 1;

/*******************************************************************************
 * Function Name: udp_server_task
 *******************************************************************************
 * Summary:
 *  Task used to establish a connection to a remote UDP client.
 *
 * Parameters:
 *  void *args : Task parameter defined during task creation (unused)
 *
 * Return:
 *  void
 *
 *******************************************************************************/
void udp_server_task(void *arg)
{
    cy_rslt_t result;

    /* Variable to store number of bytes sent over UDP socket. */
    uint32_t nbytes_sent = 0;

    /* number of bytes to send over UDP socket */
    uint32_t dataTypeToSend = 0;	// message to send is either command (0) or data (1)

    /* Initialize the user button (CYBSP_USER_BTN) and register interrupt on falling edge. */
    cyhal_gpio_init(CYBSP_USER_BTN, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_PULLUP, CYBSP_BTN_OFF);
    cyhal_gpio_register_callback(CYBSP_USER_BTN, &cb_data);
    cyhal_gpio_enable_event(CYBSP_USER_BTN, CYHAL_GPIO_IRQ_FALL, USER_BTN_INTR_PRIORITY, true);

    /* Connect to Wi-Fi AP */
    if(connect_to_wifi_ap() != CY_RSLT_SUCCESS )
    {
        printf("\n Failed to connect to Wi-Fi AP.\n");
        CY_ASSERT(0);
    }

    /* Secure Sockets initialization */
    result = cy_socket_init();
    if (result != CY_RSLT_SUCCESS)
    {
        printf("Secure Sockets initialization failed!\n");
        CY_ASSERT(0);
    }
    printf("Secure Sockets initialized\n");

    /* Create UDP Server*/
    result = create_udp_server_socket();
    if (result != CY_RSLT_SUCCESS)
    {
        printf("UDP Server Socket creation failed. Error: %"PRIu32"\n", result);
        CY_ASSERT(0);
    }

    /* Broadcast data to client */
    while(true)
    {
        /* Wait until a notification is received from the user button ISR. */
        xTaskNotifyWait(0, 0, &dataTypeToSend, portMAX_DELAY);

        /* Send LED ON/OFF command to UDP client. */
        if(dataTypeToSend == typeToSend_data && client_connected)	// radar data
        {
            result = cy_socket_sendto(server_handle, pBufToSend, szBufToSend, CY_SOCKET_FLAGS_NONE,
                                      &peer_addr, sizeof(cy_socket_sockaddr_t), &nbytes_sent);
            if(result == CY_RSLT_SUCCESS )
            {
                printf("%lu bytes data sent to %s\n", szBufToSend, peer_addr_s());
            }
            else
            {
                printf("Failed send data to %s. Err: %"PRIu32"\n", peer_addr_s(), result);
                client_connected = false;
            }
            //print_heap_usage("After sending to client");
		} else if(dataTypeToSend == typeToSend_cmd) {	// command
			result = cy_socket_sendto(server_handle, pCmdToSend, szCmdToSend, CY_SOCKET_FLAGS_NONE,
									&peer_addr_cmd, sizeof(cy_socket_sockaddr_t), &nbytes_sent);
			if(result == CY_RSLT_SUCCESS )
			{
			  printf("%lu bytes cmd sent to %s: %s\n", szCmdToSend, peer_addr_cmd_s(), pCmdToSend);
			}
			else
			{
			  printf("Failed send cmd to %s. Err: %"PRIu32"\n", peer_addr_cmd_s(), result);
			  client_connected = false;
			}
			//print_heap_usage("After sending to client");
		}
    }
 }

/*******************************************************************************
 * Function Name: connect_to_wifi_ap()
 *******************************************************************************
 * Summary:
 *  Connects to Wi-Fi AP using the user-configured credentials, retries up to a
 *  configured number of times until the connection succeeds.
 *
 *******************************************************************************/
cy_rslt_t connect_to_wifi_ap(void)
{
    cy_rslt_t result;

    /* Variables used by Wi-Fi connection manager. */
    cy_wcm_connect_params_t wifi_conn_param;

    cy_wcm_config_t wifi_config = {
            .interface = CY_WCM_INTERFACE_TYPE_STA
    };

    cy_wcm_ip_address_t ip_address;

    /* Variable to track the number of connection retries to the Wi-Fi AP specified
     * by WIFI_SSID macro */
    int conn_retries = 0;

    /* Initialize Wi-Fi connection manager. */
    result = cy_wcm_init(&wifi_config);

    if (result != CY_RSLT_SUCCESS)
    {
        printf("Wi-Fi Connection Manager initialization failed!\n");
        return result;
    }
    printf("Wi-Fi Connection Manager initialized. \n");

    /* Set the Wi-Fi SSID, password and security type. */
    memset(&wifi_conn_param, 0, sizeof(cy_wcm_connect_params_t));
    memcpy(wifi_conn_param.ap_credentials.SSID, WIFI_SSID, sizeof(WIFI_SSID));
    memcpy(wifi_conn_param.ap_credentials.password, WIFI_PASSWORD, sizeof(WIFI_PASSWORD));
    wifi_conn_param.ap_credentials.security = WIFI_SECURITY_TYPE;

    /* Join the Wi-Fi AP. */
    //for(conn_retries = 0; conn_retries < MAX_WIFI_CONN_RETRIES; conn_retries++ )
    for(conn_retries = 0; true ; conn_retries++ )
    {
        result = cy_wcm_connect_ap(&wifi_conn_param, &ip_address);

        if(result == CY_RSLT_SUCCESS)
        {
            printf("Successfully connected to Wi-Fi network '%s'.\n",
                    wifi_conn_param.ap_credentials.SSID);
            printf("IP Address Assigned: %d.%d.%d.%d\n", (uint8)ip_address.ip.v4,
                    (uint8)(ip_address.ip.v4 >> 8), (uint8)(ip_address.ip.v4 >> 16),
                    (uint8)(ip_address.ip.v4 >> 24));

            /* IP address and UDP port number of the UDP server */
            udp_server_addr.ip_address.ip.v4 = ip_address.ip.v4;
            udp_server_addr.ip_address.version = CY_SOCKET_IP_VER_V4;
            udp_server_addr.port = UDP_SERVER_PORT;
            sprintf(server_addr_s, addrToStr(&udp_server_addr));
            return result;
        }

        printf("Connection to Wi-Fi network failed with error code %d."
                "Retrying in %d ms...\n", (int)result, WIFI_CONN_RETRY_INTERVAL_MSEC);
        vTaskDelay(pdMS_TO_TICKS(WIFI_CONN_RETRY_INTERVAL_MSEC));
    }

    /* Stop retrying after maximum retry attempts. */
    printf("Exceeded maximum Wi-Fi connection attempts\n");

    return result;
}

/*******************************************************************************
 * Function Name: create_udp_server_socket
 *******************************************************************************
 * Summary:
 *  Function to create a socket and set the socket options
 *
 *******************************************************************************/
cy_rslt_t create_udp_server_socket(void)
{
    cy_rslt_t result;

    /* Variable used to set socket options. */
    cy_socket_opt_callback_t udp_recv_option = {
            .callback = udp_server_recv_handler,
            .arg = NULL
    };

    /* Create a UDP server socket. */
    result = cy_socket_create(CY_SOCKET_DOMAIN_AF_INET, CY_SOCKET_TYPE_DGRAM, CY_SOCKET_IPPROTO_UDP, &server_handle);
    if (result != CY_RSLT_SUCCESS)
    {
        return result;
    }

    /* Register the callback function to handle messages received from UDP client. */
    result = cy_socket_setsockopt(server_handle, CY_SOCKET_SOL_SOCKET,
            CY_SOCKET_SO_RECEIVE_CALLBACK,
            &udp_recv_option, sizeof(cy_socket_opt_callback_t));
    if (result != CY_RSLT_SUCCESS)
    {
        return result;
    }

    /* Bind the UDP socket created to Server IP address and port. */
    result = cy_socket_bind(server_handle, &udp_server_addr, sizeof(udp_server_addr));
    if (result == CY_RSLT_SUCCESS)
    {
         printf("Socket bound to port: %d\n", udp_server_addr.port);
    }

    return result;
}

/*******************************************************************************
 * Function Name: udp_server_recv_handler
 *******************************************************************************
 * Summary:
 *  Callback function to handle incoming  message from UDP client
 *
 *******************************************************************************/
cy_rslt_t udp_server_recv_handler(cy_socket_t socket_handle, void *arg)
{
    cy_rslt_t result;

    /* Variable to store the number of bytes received. */
    uint32_t bytes_received = 0;

    /* Buffer to store data received from Client. */
    char message_buffer[MAX_UDP_RECV_BUFFER_SIZE] = {0};

    /* Receive incoming message from UDP server. */
    result = cy_socket_recvfrom(server_handle, message_buffer, MAX_UDP_RECV_BUFFER_SIZE,
                                CY_SOCKET_FLAGS_NONE, &peer_addr, NULL,
                                &bytes_received);

    message_buffer[bytes_received] = '\0';

    if (result == CY_RSLT_SUCCESS)
    {
    	/* Print the message received from UDP client. */
    	printf("udp_server_recv_handler received cmd from client %s: %s\n",peer_addr_s(), message_buffer);

		if (strcmp(DISCOVERY_MSG, message_buffer) == 0) {
			client_connected = false;
			peer_addr_cmd = peer_addr;	// peer_addr for commands
			pCmdToSend = (uint8*)server_addr_s;
			szCmdToSend = strlen((char*)pCmdToSend);
			xTaskNotify(server_task_handle, typeToSend_cmd, eSetValueWithoutOverwrite);	// command to send

		} else if (strcmp(START_FRAME_MSG, message_buffer) == 0) {	// peer_addr for data
            client_connected = true;
            printf("UDP Client available on IP Address: %s \n", peer_addr_s());

		} else if (strcmp(END_FRAME_MSG, message_buffer) == 0) {
			client_connected = false;
			printf("UDP Client %s quits receiving \n", peer_addr_s());
		} else {
			client_connected = false;
            printf("Unexpected command. Ignored\n");
        }
    }
    else
    {
        printf("Failed to receive message from client. Error: %"PRIu32"\n", result);
        return result;
    }

    return result;
}

/*******************************************************************************
 * Function Name: isr_button_press
 *******************************************************************************
 *
 * Summary:
 *  GPIO interrupt service routine. This function detects button presses and
 *  sets the command to be sent to UDP client.
 *
 * Parameters:
 *  void *callback_arg : pointer to the variable passed to the ISR
 *  cyhal_gpio_event_t event : GPIO event type
 *
 * Return:
 *  None
 *
 *******************************************************************************/
void isr_button_press( void *callback_arg, cyhal_gpio_event_t event)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    static uint8 buf_to_send[4];

    /* Set the command to be sent to UDP client. */
    if(led_state == CYBSP_LED_STATE_ON) {
        buf_to_send[0] = LED_OFF_CMD;
    }
    else {
    	buf_to_send[0] = LED_ON_CMD;
    }
    buf_to_send[1] = '\0';

    /* Set the flag to send command to UDP client. */
    pCmdToSend = &buf_to_send[0];
    szCmdToSend = 1;
    xTaskNotifyFromISR(server_task_handle, typeToSend_cmd,
                      eSetValueWithoutOverwrite, &xHigherPriorityTaskWoken);
    /* Force a context switch if xHigherPriorityTaskWoken is now set to pdTRUE. */
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

/* [] END OF FILE */

