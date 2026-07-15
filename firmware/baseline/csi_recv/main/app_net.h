// IoT dio prijemnika: WiFi STA, MQTT i ThingSpeak
// kada je APP_NET_PUBLISH jednak nuli, funkcije ostaju prazne i ne traže dodatne #if grane
// netif i ručno postavljanje kanala ostaju uslovni jer utiču na redoslijed pokretanja WiFi-ja
#ifndef APP_NET_H
#define APP_NET_H

#include <stdbool.h>
#include <stdint.h>
#include "app_net_config.h"

// povezuje WiFi i pokreće MQTT i ThingSpeak taskove poslije WiFi i CSI sloja
void app_net_start(void);

// postojeći FreeRTOS taskovi ovim funkcijama osvježavaju telemetriju
void app_net_update_tinyml(const char *state, float person_probability,
                           float confidence, int latency_us,
                           uint32_t heap_free, uint32_t frames);
void app_net_update_dht(int temperature_c, int humidity_pct, bool valid);
void app_net_update_mmwave(bool present);
void app_net_note_csi_frame(void); // pozovi jednom za svaki obradjeni CSI frejm

#endif // APP_NET_H
