/*
 * primjer lokalnih mrežnih podešavanja
 * kopija pod nazivom app_net_config_local.h ostaje izvan Git repozitorijuma
 */
#ifndef APP_NET_CONFIG_H
#define APP_NET_CONFIG_H

#define APP_NET_PUBLISH            0

#define APP_WIFI_SSID              "YOUR_2G_WIFI_SSID"
#define APP_WIFI_PASS              "YOUR_WIFI_PASSWORD"
#define APP_MQTT_URI               "mqtt://YOUR_LAPTOP_LAN_IP:1883"
#define APP_MQTT_TOPIC             "wifi-csi/room/state"
#define APP_MQTT_PUBLISH_MS        1000

#define APP_THINGSPEAK_API_KEY     ""
#define APP_THINGSPEAK_PERIOD_MS   16000

#define APP_HTTP_SERVER            0

#endif /* APP_NET_CONFIG_H */
