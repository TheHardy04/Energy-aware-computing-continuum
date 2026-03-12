# Inventory
| Node ID | GCP VM Name   | Role              | GCP Machine Type | CPU | RAM (MB) | Region | Zone |
|---------|---------------|-------------------|------------------|-----|----------|--------|------|
| 0       | cloud-core-1  | Cloud Master      | n2-standard-4    | 4   | 16000    | europe-west4 | europe-west4-a |
| 1       | fog-gateway-1 | Regional Router   | e2-standard-2    | 2   | 8000     | europe-west9 | europe-west9-a |
| 2       | worker-edge-1 | Heavy Edge        | e2-medium        | 2   | 4000     | europe-west9 | europe-west9-b |
| 3       | worker-edge-2 | Light Edge        | e2-small         | 1   | 2000     | europe-west9 | europe-west9-c |
| 4       | worker-iot-1  | Deep Edge         | e2-micro         | 1   | 1000     | europe-west9 | europe-west9-a |
|||||||||
| 5       | fog-gateway-2 | Regional Router 2 | e2-standard-2    | 2   | 8000     | europe-west4 | europe-west4-a |
| 6       | worker-edge-3 | Heavy Edge        | e2-medium        | 2   | 4000     | europe-north2 | europe-north2-a |
| 7       | worker-edge-4 | Light Edge        | e2-small         | 1   | 2000     | europe-north2 | europe-north2-b |
| 8       | worker-iot-2  | Deep Edge         | e2-micro         | 1   | 1000     | europe-north2 | europe-north2-c |
| 9       | worker-iot-3  | Deep Edge         | e2-qmicro         | 1   | 1000     | europe-north2 | europe-north2-a |
|||||||||
| 10      | cloud-core-2  | Cloud Region 2    | n2-standard-4    | 4   | 16000    | northamerica-northeast1 | northamerica-northeast1-a |
| 11      | fog-gateway-3 | Regional Router 3 | e2-standard-2    | 2   | 8000     | northamerica-northeast1 | northamerica-northeast1-b |
| 12      | fog-gateway-4 | Regional Router 4 | e2-medium        | 2   | 4000     | northamerica-northeast1 | northamerica-northeast1-c |
| 13      | worker-edge-5 | Light Edge        | e2-small         | 1   | 2000     | northamerica-northeast1 | northamerica-northeast1-a |
| 14      | worker-edge-6 | Light Edge        | e2-small         | 1   | 2000     | northamerica-northeast1 | northamerica-northeast1-b |
| 15      | worker-iot-4  | Deep Edge         | e2-micro         | 1   | 1000     | northamerica-northeast1 | northamerica-northeast1-c |
| 16      | worker-iot-5  | Deep Edge         | e2-micro         | 1   | 1000     | northamerica-northeast1 | northamerica-northeast1-a |
| 17      | worker-iot-6  | Deep Edge         | e2-micro         | 1   | 1000     | northamerica-northeast1 | northamerica-northeast1-b |
| 18      | worker-iot-7  | Deep Edge         | e2-micro         | 1   | 1000     | northamerica-northeast1 | northamerica-northeast1-c |
| 19      | worker-iot-8  | Deep Edge         | e2-micro         | 1   | 1000     | northamerica-northeast1 | northamerica-northeast1-a |

# GCP Regions 

From [region-carbon-data](https://github.com/GoogleCloudPlatform/region-carbon-info)

| Google Cloud Region | Location | Google CFE% | Grid carbon intensity (gCO2eq/kWh) | Notes |
|---------------------|----------|-------------|-------------------------------------|-------|
| africa-south1 | Johannesburg | 15% | 657 | |
| asia-east1 | Taiwan | 17% | 439 | |
| asia-east2 | Hong Kong | 1% | 505 | |
| asia-northeast1 | Tokyo | 17% | 453 | |
| asia-northeast2 | Osaka | 46% | 296 | |
| asia-northeast3 | Seoul | 37% | 357 | |
| asia-south1 | Mumbai | 9% | 679 | |
| asia-south2 | Delhi | 29% | 532 | |
| asia-southeast1 | Singapore | 4% | 367 | |
| asia-southeast2 | Jakarta | 18% | 561 | |
| australia-southeast1 | Sydney | 34% | 498 | |
| australia-southeast2 | Melbourne | 39% | 454 | |
| europe-central2 | Warsaw | 40% | 643 | |
| europe-north1 | Finland | 98% | 39 | 🍃 Low CO2 |
| europe-north2 | Stockholm | 100% | 3 | 🍃 Low CO2 |
| europe-southwest1 | Madrid | 87% | 89 | 🍃 Low CO2 |
| europe-west1 | Belgium | 84% | 103 | 🍃 Low CO2 |
| europe-west2 | London | 79% | 106 | 🍃 Low CO2 |
| europe-west3 | Frankfurt | 68% | 276 | |
| europe-west4 | Eemshaven | 83% | 209 | 🍃 Low CO2 |
| europe-west6 | Zürich | 98% | 15 | 🍃 Low CO2 |
| europe-west8 | Milan | 73% | 202 | |
| europe-west9 | Paris | 96% | 16 | 🍃 Low CO2 |
| europe-west10 | Berlin | 68% | 276 | |
| europe-west12 | Turin | 73% | 202 | |
| me-central1 | Doha | 1% | 366 | |
| me-central2 | Dammam | 1% | 382 | |
| me-west1 | Tel Aviv | 7% | 434 | |
| northamerica-northeast1 | Montréal | 99% | 5 | 🍃 Low CO2 |
| northamerica-northeast2 | Toronto | 84% | 59 | 🍃 Low CO2 |
| northamerica-south1 | Mexico | 19% | 305 | |
| southamerica-east1 | Sāo Paulo | 88% | 67 | 🍃 Low CO2 |
| southamerica-west1 | Santiago | 92% | 238 | 🍃 Low CO2 |
| us-central1 | Iowa | 87% | 413 | 🍃 Low CO2 |
| us-central2 | Iowa | 88% | 372 | 🍃 Low CO2 |
| us-east1 | South Carolina | 31% | 576 | |
| us-east2 | Georgia | 42% | 340 | |
| us-east4 | Northern Virginia | 62% | 323 | |
| us-east5 | Columbus | 62% | 323 | |
| us-south1 | Dallas | 94% | 303 | 🍃 Low CO2 |
| us-west1 | Oregon | 87% | 79 | 🍃 Low CO2 |
| us-west2 | Los Angeles | 63% | 169 | |
| us-west3 | Salt Lake City | 33% | 555 | |
| us-west4 | Las Vegas | 64% | 357 | |