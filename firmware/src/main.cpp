// Offline, direct-answer AI on an ESP32. Type a question over serial:
//
//   you: how much does a hummingbird weigh
//   esp: Between 2 and 20 grams.
#include <Arduino.h>

#include "model_data.h"
#include "tinyai.h"

#define SHOW_TIMING 0  // 1 = print how long each answer took

static tai_model model;
static char line[256];
static size_t line_len = 0;

void setup() {
    Serial.begin(115200);
    delay(500);
    int rc = tai_load(&model, model_data, model_data_len);
    if (rc) {
        Serial.printf("model load failed (%d)\n", rc);
        for (;;) delay(1000);
    }
    Serial.printf("\ntinyai ready: %d facts, %u KB model, %u KB heap free\n", model.n_facts,
                  model_data_len / 1024, (unsigned)(ESP.getFreeHeap() / 1024));
    Serial.print("you: ");
}

void loop() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\r' || c == '\n') {
            if (!line_len) continue;
            line[line_len] = 0;
            line_len = 0;
            char out[64];
            uint32_t t0 = millis();
            tai_ask(&model, line, out, sizeof out);
            uint32_t ms = millis() - t0;
            Serial.printf("esp: %s\n", out);
            if (SHOW_TIMING) Serial.printf("     (%u ms)\n", (unsigned)ms);
            Serial.print("you: ");
        } else if (line_len < sizeof line - 1) {
            line[line_len++] = c;
        }
    }
}
