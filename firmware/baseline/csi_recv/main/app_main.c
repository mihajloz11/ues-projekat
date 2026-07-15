// prijemnik je zasnovan na Espressif CSI get-started primjeru
// obrađuje ESP-NOW CSI, DHT11 i mmWave podatke i pokreće TinyML model na pločici
// rezultati se šalju na serijski port kao JSON linije

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "nvs_flash.h"

#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_rom_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "tinyml_presence_runtime.h"
#include "app_net.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 || (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT40
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   0
#define MMWAVE_OT2_GPIO                     GPIO_NUM_5
#define MMWAVE_UART_ENABLE                  1
#define MMWAVE_UART_NUM                     UART_NUM_1
#define MMWAVE_UART_RX_GPIO                 GPIO_NUM_18
#define MMWAVE_UART_BAUD                    115200
#define MMWAVE_UART_BUF_SIZE               1024
#define CONFIG_CSI_PRINT_EVERY_N            2
#define CSI_QUEUE_LENGTH                    12
#define CSI_MAX_DATA_LEN                    612

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF                      0
#endif

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL                 1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

typedef struct {
    uint32_t rx_id;
    uint8_t mac[6];
    wifi_pkt_rx_ctrl_t rx_ctrl;
    uint16_t len;
    uint8_t first_word_invalid;
    int8_t buf[CSI_MAX_DATA_LEN];
} csi_frame_t;

static QueueHandle_t s_csi_queue;
static volatile uint32_t s_csi_queue_dropped;

// vjerovatnoća 0..1 se pretvara u cijeli broj 0..1000 radi lakšeg ispisa
static int probability_milli(float value)
{
    if (value < 0.0f) {
        value = 0.0f;
    } else if (value > 1.0f) {
        value = 1.0f;
    }
    return (int)(value * 1000.0f + 0.5f);
}

// čeka traženi nivo pina i vraća trajanje ili -1 poslije isteka vremena
static int dht11_wait_for_level(gpio_num_t gpio, int level, uint32_t timeout_us)
{
    uint32_t waited = 0;

    while (gpio_get_level(gpio) != level) {
        if (waited >= timeout_us) {
            return -1;
        }
        esp_rom_delay_us(1);
        waited++;
    }

    return waited;
}

// čita jedno DHT11 mjerenje; nula označava uspjeh, a negativna vrijednost grešku
static int dht11_read(gpio_num_t gpio, int *temperature_c, int *humidity_pct)
{
    uint8_t data[5] = {0};

    // početni signal kratko spušta liniju, poslije čega senzor odgovara
    gpio_set_direction(gpio, GPIO_MODE_OUTPUT);
    gpio_set_level(gpio, 1);
    vTaskDelay(pdMS_TO_TICKS(20));

    gpio_set_level(gpio, 0);
    vTaskDelay(pdMS_TO_TICKS(20));

    gpio_set_level(gpio, 1);
    esp_rom_delay_us(40);

    gpio_set_direction(gpio, GPIO_MODE_INPUT);
    gpio_set_pull_mode(gpio, GPIO_PULLUP_ONLY);

    if (dht11_wait_for_level(gpio, 0, 100) < 0) {
        return -1;
    }
    if (dht11_wait_for_level(gpio, 1, 100) < 0) {
        return -2;
    }
    if (dht11_wait_for_level(gpio, 0, 100) < 0) {
        return -3;
    }

    // senzor šalje 40 bita, a dužina visokog impulsa određuje vrijednost bita
    for (int bit = 0; bit < 40; bit++) {
        if (dht11_wait_for_level(gpio, 1, 70) < 0) {
            return -4;
        }

        int high_us = dht11_wait_for_level(gpio, 0, 100);
        if (high_us < 0) {
            return -5;
        }

        data[bit / 8] <<= 1;
        if (high_us > 40) {
            data[bit / 8] |= 1;
        }
    }

    // posljednji bajt je kontrolna suma prva četiri bajta
    uint8_t checksum = data[0] + data[1] + data[2] + data[3];
    if (checksum != data[4]) {
        return -6;
    }

    *humidity_pct = data[0];
    *temperature_c = data[2];
    return 0;
}

// task svake dvije sekunde čita DHT11 i mmWave i ispisuje SENSOR_DATA
static void dht11_task(void *arg)
{
    (void)arg;
    int fail_count = 0;

    ESP_LOGI(TAG, "DHT11 enabled: VCC->3V3, GND->GND, DOUT->GPIO4");
    ESP_LOGI(TAG, "mmWave OT2 enabled: OT2->GPIO5, present=high");

    gpio_set_direction(MMWAVE_OT2_GPIO, GPIO_MODE_INPUT);
    gpio_set_pull_mode(MMWAVE_OT2_GPIO, GPIO_PULLDOWN_ONLY);

    vTaskDelay(pdMS_TO_TICKS(1000));

    while (1) {
        int mmwave_present = gpio_get_level(MMWAVE_OT2_GPIO) ? 1 : 0;

        int temperature_c = 0;
        int humidity_pct = 0;
        int rc = dht11_read(GPIO_NUM_4, &temperature_c, &humidity_pct);

        ets_printf("SENSOR_DATA,{\"sensor\":\"mmwave\",\"present\":%s,\"source\":\"OT2\",\"gpio\":5}\n",
                   mmwave_present ? "true" : "false");
        app_net_update_mmwave(mmwave_present);

        if (rc == 0) {
            fail_count = 0;
            ets_printf("SENSOR_DATA,{\"sensor\":\"dht11\",\"temperature_c\":%d,\"humidity_pct\":%d,\"gpio\":4}\n",
                       temperature_c, humidity_pct);
            app_net_update_dht(temperature_c, humidity_pct, true);
        } else {
            fail_count++;
            if (fail_count == 1 || fail_count % 10 == 0) {
                ets_printf("SENSOR_DATA,{\"sensor\":\"dht11\",\"status\":\"read_failed\",\"error\":%d,\"fail_count\":%d,\"gpio\":4}\n",
                           rc, fail_count);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}

#if MMWAVE_UART_ENABLE
// čita ASCII linije sa 24 GHz mmWave modula i prosljeđuje ih kao MMWAVE_TXT
// veza je jednosmjerna: mmWave TX na GPIO18, uz zajedničke 3V3 i GND; OT2 ostaje na GPIO5
static void mmwave_uart_task(void *arg)
{
    (void)arg;
    const uart_config_t uart_config = {
        .baud_rate  = MMWAVE_UART_BAUD,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    if (uart_driver_install(MMWAVE_UART_NUM, MMWAVE_UART_BUF_SIZE * 2, 0, 0, NULL, 0) != ESP_OK
            || uart_param_config(MMWAVE_UART_NUM, &uart_config) != ESP_OK
            || uart_set_pin(MMWAVE_UART_NUM, UART_PIN_NO_CHANGE, MMWAVE_UART_RX_GPIO,
                            UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE) != ESP_OK) {
        ESP_LOGE(TAG, "mmWave UART init failed");
        vTaskDelete(NULL);
        return;
    }

    // pull-up sprečava lažne bajtove kada RX pin nije povezan
    gpio_set_pull_mode(MMWAVE_UART_RX_GPIO, GPIO_PULLUP_ONLY);

    ESP_LOGI(TAG, "mmWave UART line forwarder on GPIO%d at %d baud", MMWAVE_UART_RX_GPIO, MMWAVE_UART_BAUD);

    // završene linije poput "ON", "OFF" i "Range NNN" prosljeđuju se kao MMWAVE_TXT
    static uint8_t data[64];
    static char linebuf[128];
    int pos = 0;

    while (1) {
        int len = uart_read_bytes(MMWAVE_UART_NUM, data, sizeof(data), pdMS_TO_TICKS(100));
        for (int i = 0; i < len; i++) {
            char c = (char)data[i];
            if (c == '\r') {
                continue;
            }
            if (c == '\n') {
                if (pos > 0) {
                    linebuf[pos] = '\0';
                    ets_printf("MMWAVE_TXT,%s\n", linebuf);
                    pos = 0;
                }
            } else if (pos < (int)sizeof(linebuf) - 1) {
                linebuf[pos++] = c;
            } else {
                pos = 0;
            }
        }
    }
}
#endif

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
#if APP_NET_PUBLISH
    // STA netif daje prijemniku IP pristup za MQTT i ThingSpeak
    // radio tada prati 2.4 GHz ruter, koji mora biti na kanalu 11 kao i pošiljalac
    esp_netif_create_default_wifi_sta();
#endif
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
        .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
#if CONFIG_IDF_TARGET_ESP32C5
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20)
            || (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY && CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20)) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    // ESP-NOW HT40 traži podešen primarni i sekundarni kanal
    // uz APP_NET_PUBLISH kasniji esp_wifi_connect() prebacuje radio na kanal rutera
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif

    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));

}

// task preuzima frejmove iz reda, ispisuje CSI liniju i pokreće TinyML model
static void csi_processing_task(void *arg)
{
    (void)arg;
    csi_frame_t frame;
    uint32_t processed_count = 0;
    uint32_t last_reported_dropped = 0;
    uint8_t agc_gain = 0;
    int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    uint8_t agc_gain_baseline = 0;
    int8_t fft_gain_baseline = 0;
#endif

    while (1) {
        if (xQueueReceive(s_csi_queue, &frame, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        app_net_note_csi_frame();

        const wifi_pkt_rx_ctrl_t *rx_ctrl = &frame.rx_ctrl;
        float compensate_gain = 1.0f;
#if CONFIG_GAIN_CONTROL
        esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
        if (processed_count < 100) {
            esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
        } else if (processed_count == 100) {
            esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
            esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
            ESP_LOGD(TAG, "fft_force %d, agc_force %d", fft_gain_baseline, agc_gain_baseline);
#endif
        }
        esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
        ESP_LOGD(TAG, "compensate_gain %f, agc_gain %d, fft_gain %d", compensate_gain, agc_gain, fft_gain);
#endif

        uint32_t dropped = s_csi_queue_dropped;
        if (dropped != last_reported_dropped) {
            ESP_LOGW(TAG, "CSI processing queue dropped %lu frames", dropped);
            last_reported_dropped = dropped;
        }

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
        if (!processed_count) {
            ESP_LOGI(TAG, "================ CSI RECV ================");
            ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_format,len,first_word,data\n");
        }

        ets_printf("CSI_DATA,%lu," MACSTR ",%d,%d,%d,%d,%d,%d,%lu,%d,%d",
                   frame.rx_id, MAC2STR(frame.mac), rx_ctrl->rssi, rx_ctrl->rate,
                   rx_ctrl->noise_floor, fft_gain, agc_gain,  rx_ctrl->channel,
                   rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->cur_bb_format);
#else
        if (!processed_count) {
            ESP_LOGI(TAG, "================ CSI RECV ================");
            ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data\n");
        }

        ets_printf("CSI_DATA,%lu," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lu,%d,%d",
                   frame.rx_id, MAC2STR(frame.mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
                   rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
                   rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
                   rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
                   rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->sig_mode);

#endif
#if (CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61) && CSI_FORCE_LLTF
        int16_t csi = ((int16_t)(((((uint16_t)frame.buf[1]) << 8) | frame.buf[0]) << 4) >> 4);
        ets_printf(",%d,%d,\"[%d", (frame.len - 2) / 2, frame.first_word_invalid, (int16_t)(compensate_gain * csi));
        for (int i = 2; i < (frame.len - 2); i += 2) {
            csi = ((int16_t)(((((uint16_t)frame.buf[i + 1]) << 8) | frame.buf[i]) << 4) >> 4);
            ets_printf(",%d", (int16_t)(compensate_gain * csi));
        }
#else
        ets_printf(",%u,%d,\"[%d", frame.len, frame.first_word_invalid, (int16_t)(compensate_gain * frame.buf[0]));
        for (int i = 1; i < frame.len; i++) {
            ets_printf(",%d", (int16_t)(compensate_gain * frame.buf[i]));
        }
#endif
        ets_printf("]\"\n");

        tinyml_presence_result_t tinyml_result;
        int tinyml_latency_us = 0;
        uint32_t tinyml_frames = 0;
        if (tinyml_presence_runtime_push_iq(frame.buf,
                                            frame.len,
                                            compensate_gain,
                                            &tinyml_result,
                                            &tinyml_latency_us,
                                            &tinyml_frames)) {
            ets_printf("TINYML_DATA,{\"model\":\"presence_csi_mlp_fast_int8\",\"source\":\"esp32s3\",\"state\":\"%s\",\"person_probability_milli\":%d,\"confidence_milli\":%d,\"latency_us\":%d,\"heap_free\":%lu,\"window\":%d,\"frames\":%lu,\"queue_dropped\":%lu}\n",
                       tinyml_result.label,
                       probability_milli(tinyml_result.person_probability),
                       probability_milli(tinyml_result.confidence),
                       tinyml_latency_us,
                       esp_get_free_heap_size(),
                       TINYML_PRESENCE_WINDOW_SIZE,
                       tinyml_frames,
                       s_csi_queue_dropped);

            app_net_update_tinyml(tinyml_result.label,
                                  tinyml_result.person_probability,
                                  tinyml_result.confidence,
                                  tinyml_latency_us,
                                  esp_get_free_heap_size(),
                                  tinyml_frames);
        }
        processed_count++;
    }
}

// WiFi callback samo kopira CSI frejm u red, a obradu radi csi_processing_task
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    static uint32_t s_seen_count = 0;
    static csi_frame_t frame;

    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }

    uint32_t frame_index = s_seen_count++;
    if ((frame_index % CONFIG_CSI_PRINT_EVERY_N) != 0) {
        return;
    }

    if (s_csi_queue == NULL || info->len > CSI_MAX_DATA_LEN) {
        s_csi_queue_dropped++;
        return;
    }

    frame.rx_id = *(uint32_t *)(info->payload + 15);
    memcpy(frame.mac, info->mac, sizeof(frame.mac));
    frame.rx_ctrl = info->rx_ctrl;
    frame.len = (uint16_t)info->len;
    frame.first_word_invalid = (uint8_t)info->first_word_invalid;
    memcpy(frame.buf, info->buf, frame.len);

    if (xQueueSend(s_csi_queue, &frame, 0) != pdTRUE) {
        s_csi_queue_dropped++;
    }
}

// priprema red i task za obradu, zatim uključuje CSI i prijavljuje callback
static void wifi_csi_init()
{
    s_csi_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(csi_frame_t));
    if (s_csi_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create CSI processing queue");
        abort();
    }
    xTaskCreate(csi_processing_task, "csi_process", 12288, NULL, 18, NULL);

    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    // podrazumijevana CSI konfiguracija zavisi od čipa
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable                   = true,
        .acquire_csi_legacy       = false,
        .acquire_csi_force_lltf   = CSI_FORCE_LLTF,
        .acquire_csi_ht20         = true,
        .acquire_csi_ht40         = true,
        .acquire_csi_vht          = false,
        .acquire_csi_su           = false,
        .acquire_csi_mu           = false,
        .acquire_csi_dcm          = false,
        .acquire_csi_beamformed   = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg            = 0,
        .dump_ack_en              = false,
        .reserved                 = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = false,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = true,
        .acquire_csi_mu         = true,
        .acquire_csi_dcm        = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main()
{
    // NVS se pokreće prvi jer ga WiFi koristi
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // pokretanje WiFi sloja
    wifi_init();

    // ESP-NOW broadcast peer prima pakete pošiljaoca na istom kanalu
    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };

    wifi_esp_now_init(peer);

    xTaskCreate(dht11_task, "dht11_task", 4096, NULL, 20, NULL);

#if MMWAVE_UART_ENABLE
    xTaskCreate(mmwave_uart_task, "mmwave_uart", 4096, NULL, 10, NULL);
#endif

    wifi_csi_init();

    // IoT publisher se pokreće poslije CSI i ESP-NOW sloja
    // kada je APP_NET_PUBLISH jednak nuli, funkcija ostaje prazna
    app_net_start();
}
