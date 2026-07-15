/*
 * javne podrazumijevane vrijednosti za ESP32-S3 IoT sloj
 *
 * WiFi lozinke i cloud ključevi pripadaju lokalnom app_net_config_local.h fajlu,
 * napravljenom prema app_net_config_local.example.h primjeru
 */
#if __has_include("app_net_config_local.h")
#include "app_net_config_local.h"
#else
#ifndef APP_NET_CONFIG_H
#define APP_NET_CONFIG_H

/* 0 = gateway/serijski mod sa CSI demonstracijom
 * 1 = ESP32-S3 se povezuje na WiFi i šalje telemetriju na MQTT i ThingSpeak */
#define APP_NET_PUBLISH            0

/* javne zamjenske vrijednosti; lokalna podešavanja su u app_net_config_local.h */
#define APP_WIFI_SSID              "CHANGE_ME_2G_WIFI"
#define APP_WIFI_PASS              "CHANGE_ME_WIFI_PASSWORD"
#define APP_MQTT_URI               "mqtt://192.168.1.100:1883"
#define APP_MQTT_TOPIC             "wifi-csi/room/state"
#define APP_MQTT_PUBLISH_MS        1000

#define APP_THINGSPEAK_API_KEY     ""
#define APP_THINGSPEAK_PERIOD_MS   16000

/* opcioni HTTP server za prikaz stanja na uređaju */
#define APP_HTTP_SERVER            0

#endif /* APP_NET_CONFIG_H */
#endif /* lokalna podešavanja */
