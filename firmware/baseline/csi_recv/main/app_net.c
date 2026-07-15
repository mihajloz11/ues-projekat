// IoT publisher: WiFi STA, MQTT i ThingSpeak
// sloj je odvojen od CSI, TinyML i serijskog puta u app_main.c
#include "app_net.h"

#include <string.h>
#include <stdio.h>

#if APP_NET_PUBLISH

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "mqtt_client.h"
#include "esp_http_client.h"

static const char *TAG = "app_net";

#define NET_CONNECTED_BIT BIT0

// zajedničku telemetriju upisuju taskovi, a čita publisher
typedef struct {
    char state[24];
    char edge_state[24];
    float confidence;
    float person_prob;
    int latency_us;
    uint32_t heap_free;
    uint32_t frames;
    int temperature_c;
    int humidity_pct;
    bool dht_valid;
    bool mmwave_present;
} telemetry_t;

static telemetry_t s_tel = {
    .state = "warming_up",
    .edge_state = "warming_up",
};
static portMUX_TYPE s_tel_mux = portMUX_INITIALIZER_UNLOCKED;

static volatile uint32_t s_csi_count;
static volatile int64_t s_last_csi_us;

static EventGroupHandle_t s_net_events;
static esp_mqtt_client_handle_t s_mqtt;
static volatile bool s_mqtt_online;

// setteri se pozivaju iz više taskova, pa pristup ide kroz kritičnu sekciju
void app_net_update_tinyml(const char *state, float person_probability,
                           float confidence, int latency_us,
                           uint32_t heap_free, uint32_t frames)
{
    portENTER_CRITICAL(&s_tel_mux);
    strncpy(s_tel.state, state, sizeof(s_tel.state) - 1);
    s_tel.state[sizeof(s_tel.state) - 1] = '\0';
    strncpy(s_tel.edge_state, state, sizeof(s_tel.edge_state) - 1);
    s_tel.edge_state[sizeof(s_tel.edge_state) - 1] = '\0';
    s_tel.person_prob = person_probability;
    s_tel.confidence = confidence;
    s_tel.latency_us = latency_us;
    s_tel.heap_free = heap_free;
    s_tel.frames = frames;
    portEXIT_CRITICAL(&s_tel_mux);
}

void app_net_update_dht(int temperature_c, int humidity_pct, bool valid)
{
    portENTER_CRITICAL(&s_tel_mux);
    if (valid) {
        s_tel.temperature_c = temperature_c;
        s_tel.humidity_pct = humidity_pct;
    }
    s_tel.dht_valid = valid;
    portEXIT_CRITICAL(&s_tel_mux);
}

void app_net_update_mmwave(bool present)
{
    portENTER_CRITICAL(&s_tel_mux);
    s_tel.mmwave_present = present;
    portEXIT_CRITICAL(&s_tel_mux);
}

void app_net_note_csi_frame(void)
{
    s_csi_count++;
    s_last_csi_us = esp_timer_get_time();
}

// WiFi događaji ponavljaju vezu poslije prekida i provjeravaju kanal nakon dobijanja IP adrese
static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *d = (wifi_event_sta_disconnected_t *)data;
        xEventGroupClearBits(s_net_events, NET_CONNECTED_BIT);
        ESP_LOGW(TAG, "WiFi disconnected (reason=%d), retrying...", d ? d->reason : -1);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *evt = (ip_event_got_ip_t *)data;
        uint8_t primary = 0;
        wifi_second_chan_t second = 0;
        esp_wifi_get_channel(&primary, &second);
        wifi_bandwidth_t bw = WIFI_BW_HT20;
        esp_wifi_get_bandwidth(WIFI_IF_STA, &bw);
        ESP_LOGI(TAG, "Got IP " IPSTR " on WiFi channel %d, second=%d, bandwidth=%s",
                 IP2STR(&evt->ip_info.ip), primary, second,
                 bw == WIFI_BW_HT40 ? "HT40(40MHz)" : "HT20(20MHz)");
        if (primary != 11) {
            ESP_LOGW(TAG, "ROUTER IS ON CHANNEL %d, NOT 11 — CSI from the sender will be lost! "
                          "The 2.4 GHz router must use channel 11.", primary);
        }
        xEventGroupSetBits(s_net_events, NET_CONNECTED_BIT);
    }
}

// MQTT događaji samo bilježe stanje veze
static void mqtt_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    switch ((esp_mqtt_event_id_t)id) {
    case MQTT_EVENT_CONNECTED:
        s_mqtt_online = true;
        ESP_LOGI(TAG, "MQTT connected to %s", APP_MQTT_URI);
        break;
    case MQTT_EVENT_DISCONNECTED:
        s_mqtt_online = false;
        ESP_LOGW(TAG, "MQTT disconnected");
        break;
    default:
        break;
    }
}

// šalje odabrana polja na ThingSpeak običnim HTTP GET zahtjevom
#if defined(APP_THINGSPEAK_API_KEY)
static void thingspeak_post(const telemetry_t *t, bool csi_online)
{
    if (APP_THINGSPEAK_API_KEY[0] == '\0') {
        return; // cloud iskljucen
    }
    char url[320];
    int present = (strcmp(t->state, "person_present") == 0) ? 1 : 0;
    snprintf(url, sizeof(url),
             "http://api.thingspeak.com/update?api_key=%s"
             "&field1=%d&field2=%.3f&field3=%d&field4=%d&field5=%d&field6=%d",
             APP_THINGSPEAK_API_KEY, present, t->confidence,
             t->temperature_c, t->humidity_pct,
             t->mmwave_present ? 1 : 0, t->latency_us);

    esp_http_client_config_t cfg = {
        .url = url,
        .timeout_ms = 8000,
    };
    esp_http_client_handle_t c = esp_http_client_init(&cfg);
    esp_err_t err = esp_http_client_perform(c);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "ThingSpeak update -> HTTP %d", esp_http_client_get_status_code(c));
    } else {
        ESP_LOGW(TAG, "ThingSpeak post failed: %s", esp_err_to_name(err));
    }
    esp_http_client_cleanup(c);
}
#endif

// glavni task periodično sastavlja JSON i šalje ga na MQTT i ThingSpeak
static void net_publish_task(void *arg)
{
    (void)arg;
    char json[480];
    uint32_t prev_count = 0;
    int64_t prev_us = esp_timer_get_time();
    int64_t last_ts_post = 0;
    bool reasserted = false;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(APP_MQTT_PUBLISH_MS));

        // veza sa 20 MHz ruterom može izbaciti radio iz HT40 i ugasiti promiscuous mod
        // poslije povezivanja oba se vraćaju da CSI nastavi raditi uz MQTT vezu
        if (!reasserted && (xEventGroupGetBits(s_net_events) & NET_CONNECTED_BIT)) {
            esp_err_t eb = esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40);
            esp_err_t ep = esp_wifi_set_promiscuous(true);
            ESP_LOGI(TAG, "post-connect reassert: HT40=%s promisc=%s",
                     esp_err_to_name(eb), esp_err_to_name(ep));
            reasserted = true;
        }

        // trenutna telemetrija se kopira unutar kritične sekcije
        telemetry_t t;
        portENTER_CRITICAL(&s_tel_mux);
        t = s_tel;
        portEXIT_CRITICAL(&s_tel_mux);

        int64_t now = esp_timer_get_time();
        uint32_t count = s_csi_count;
        float dt = (now - prev_us) / 1e6f;
        float fps = (dt > 0.0f) ? ((count - prev_count) / dt) : 0.0f;
        prev_count = count;
        prev_us = now;
        bool csi_online = (now - s_last_csi_us) < 2000000; // manje od 2 s od zadnjeg frejma

        // u device-direct modu radio prelazi sa HT40 na 20 MHz i CSI prestaje raditi
        // tada mmWave daje stanje prisustva, dok gateway mod zadržava TinyML rezultat
        const char *pub_state = t.state;
        float pub_conf = t.confidence;
        if (!csi_online) {
            pub_state = t.mmwave_present ? "person_present" : "empty_room";
            pub_conf = 1.0f;
        }

        int n = snprintf(json, sizeof(json),
            "{\"state\":\"%s\",\"confidence\":%.3f,"
            "\"temperature_c\":%d,\"humidity_pct\":%d,\"mmwave_present\":%s,"
            "\"csi\":{\"status\":\"%s\",\"fps\":%.1f,\"frame_count\":%lu},"
            "\"edge_tinyml\":{\"state\":\"%s\",\"person_probability\":%.3f,"
            "\"confidence\":%.3f,\"latency_us\":%d,\"heap_free\":%lu,\"frames\":%lu},"
            "\"mqtt\":{\"status\":\"online\",\"topic\":\"%s\"},"
            "\"source\":\"esp32s3\"}",
            pub_state, pub_conf,
            t.temperature_c, t.humidity_pct, t.mmwave_present ? "true" : "false",
            csi_online ? "online" : "offline", fps, (unsigned long)count,
            t.edge_state, t.person_prob, t.confidence, t.latency_us,
            (unsigned long)t.heap_free, (unsigned long)t.frames,
            APP_MQTT_TOPIC);

        if (s_mqtt && s_mqtt_online && n > 0 && n < (int)sizeof(json)) {
            esp_mqtt_client_publish(s_mqtt, APP_MQTT_TOPIC, json, n, 0, 1 /* retain */);
        }

#if defined(APP_THINGSPEAK_API_KEY)
        if ((now - last_ts_post) >= ((int64_t)APP_THINGSPEAK_PERIOD_MS * 1000)
                && (xEventGroupGetBits(s_net_events) & NET_CONNECTED_BIT)) {
            thingspeak_post(&t, csi_online);
            last_ts_post = now;
        }
#endif
    }
}

// povezuje WiFi i pokreće MQTT klijent i publisher task
void app_net_start(void)
{
    s_net_events = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = { 0 };
    strncpy((char *)wifi_config.sta.ssid, APP_WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, APP_WIFI_PASS, sizeof(wifi_config.sta.password) - 1);
    // STA prati kanal rutera, koji za ovaj raspored mora biti kanal 11
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));

    ESP_LOGI(TAG, "Connecting to SSID \"%s\" ...", APP_WIFI_SSID);
    esp_wifi_connect();

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = APP_MQTT_URI,
    };
    s_mqtt = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(s_mqtt, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(s_mqtt);

    xTaskCreate(net_publish_task, "net_publish", 5120, NULL, 5, NULL);
}

#else // APP_NET_PUBLISH == 0: prazne verzije zadržavaju isto povezivanje modula

void app_net_start(void) {}
void app_net_update_tinyml(const char *s, float a, float b, int c, uint32_t d, uint32_t e)
{ (void)s; (void)a; (void)b; (void)c; (void)d; (void)e; }
void app_net_update_dht(int a, int b, bool c) { (void)a; (void)b; (void)c; }
void app_net_update_mmwave(bool a) { (void)a; }
void app_net_note_csi_frame(void) {}

#endif
