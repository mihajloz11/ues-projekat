#include <stdint.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define DHT11_GPIO GPIO_NUM_4

static const char *TAG = "dht11_test";

static int wait_for_level(gpio_num_t gpio, int level, uint32_t timeout_us)
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

static int read_dht11(gpio_num_t gpio, int *temperature_c, int *humidity_pct)
{
    uint8_t data[5] = {0};

    gpio_set_direction(gpio, GPIO_MODE_OUTPUT);
    gpio_set_level(gpio, 1);
    vTaskDelay(pdMS_TO_TICKS(20));

    gpio_set_level(gpio, 0);
    vTaskDelay(pdMS_TO_TICKS(20));

    gpio_set_level(gpio, 1);
    esp_rom_delay_us(40);

    gpio_set_direction(gpio, GPIO_MODE_INPUT);
    gpio_set_pull_mode(gpio, GPIO_PULLUP_ONLY);

    if (wait_for_level(gpio, 0, 100) < 0) {
        return -1;
    }
    if (wait_for_level(gpio, 1, 100) < 0) {
        return -2;
    }
    if (wait_for_level(gpio, 0, 100) < 0) {
        return -3;
    }

    for (int bit = 0; bit < 40; bit++) {
        if (wait_for_level(gpio, 1, 70) < 0) {
            return -4;
        }

        int high_us = wait_for_level(gpio, 0, 100);
        if (high_us < 0) {
            return -5;
        }

        data[bit / 8] <<= 1;
        if (high_us > 40) {
            data[bit / 8] |= 1;
        }
    }

    uint8_t checksum = data[0] + data[1] + data[2] + data[3];
    if (checksum != data[4]) {
        return -6;
    }

    *humidity_pct = data[0];
    *temperature_c = data[2];
    return 0;
}

void app_main(void)
{
    ESP_LOGI(TAG, "DHT11 test firmware started");
    ESP_LOGI(TAG, "Wiring: DHT11 VCC->3V3, GND->GND, DOUT->GPIO4");

    while (1) {
        int temperature_c = 0;
        int humidity_pct = 0;
        int rc = read_dht11(DHT11_GPIO, &temperature_c, &humidity_pct);

        if (rc == 0) {
            printf("{\"sensor\":\"dht11\",\"temperature_c\":%d,\"humidity_pct\":%d}\n",
                   temperature_c, humidity_pct);
        } else {
            ESP_LOGW(TAG, "DHT11 read failed, code=%d. Expected wiring: VCC/GND/DOUT on GPIO4.", rc);
        }

        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
