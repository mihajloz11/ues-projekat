# UES projekat

- Umreženi embedded sistemi
- Mihajlo Živković
- E180/2024

Projekat koristi ESP32 kao pošiljalac WiFi paketa i ESP32-S3 kao prijemnik CSI podataka. Na ESP32-S3 pločici se izvršava TinyML model za prepoznavanje prisustva osobe u prostoriji, dok DHT11 i mmWave senzori daju dodatne podatke o stanju prostorije.

IoT dio sistema obuhvata MQTT, Node-RED, SQLite i web dashboard. Detekcija prisustva je glavni rezultat projekta, dok je procjena zone kretanja eksperimentalna.

## Struktura projekta

- `firmware` – programi za ESP32 pošiljalac, ESP32-S3 prijemnik i DHT11 test.
- `scripts` – alati za snimanje CSI podataka, treniranje i provjeru modela i pokretanje IoT sistema.
- `data` – snimljene CSI sesije i obrađeni sažeci podataka.
- `models` – modeli za prisustvo, kretanje i grubu procjenu zone.
- `iot` – Node-RED tokovi, MQTT primjer i web interfejsi.
- `outputs` – rezultati evaluacije, grafikoni i tehničke šeme.
