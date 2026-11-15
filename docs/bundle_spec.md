# Especificação do Pacote de Captura — `lidup_bundle_v1`

Contrato único entre o app iOS (`ios/LidupScanner`) e o pipeline de processamento em Python (`processor/`).

## Estrutura de Diretórios

```text
<session_id>/
├── manifest.json          # Metadados da sessão, aparelho, CRS alvo e configuração
├── frames.csv             # Índice dos keyframes capturados (pose ARKit, intrínsecas, nitidez, arquivo)
├── gnss.csv               # Leituras contínuas de GNSS (CoreLocation ou NMEA/RTK externo)
├── gcps.json              # Estacas, vértices e pontos de controle (com média estacionária + trena)
├── imu.csv                # Opcional: leituras brutas de acelerômetro e giroscópio
├── images/
│   ├── 000000.jpg         # Foto em alta resolução (JPEG qualidade >= 0.95 ou HEIC)
│   └── ...
├── depth/
│   ├── 000000.png         # PNG 16-bit em milímetros (0 = inválido), alinhado ao frame
│   └── ...
└── confidence/
    ├── 000000.png         # PNG 8-bit: 0 = low, 1 = medium, 2 = high
    └── ...
```

---

## 1. `manifest.json`

```json
{
  "schema_version": "lidup_bundle_v1",
  "session_id": "2026-11-15_113000_galpao_norte",
  "sector_name": "galpao_norte",
  "created_at_iso": "2026-11-15T14:30:00Z",
  "device": {
    "model": "iPhone16,2",
    "system_version": "18.1",
    "has_lidar": true
  },
  "capture_config": {
    "min_translation_m": 0.8,
    "min_rotation_rad": 0.21,
    "video_stabilization_disabled": true,
    "focus_locked": true,
    "image_width": 4032,
    "image_height": 3024,
    "depth_width": 256,
    "depth_height": 192,
    "depth_unit": "millimeters_uint16"
  },
  "gnss_source": "apple_core_location"
}
```

---

## 2. `frames.csv`

Convenção de pose: mesma convenção adotada no `StrayScanner` (`T_WC` — câmera para mundo local do ARKit, com `q_AC` aplicada para alinhar eixos da câmera ao padrão OpenCV/Open3D: `+X` direita, `+Y` baixo, `+Z` frente).

Colunas:
```csv
frame_id,timestamp_s,image_file,depth_file,confidence_file,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,sharpness,tracking_state,lat,lon,alt_m,h_acc_m,v_acc_m
```

* `fx, fy, cx, cy`: correspondem à resolução da imagem em `images/` (o processador escala proporcionalmente para `depth_width × depth_height` ao projetar o LiDAR).
* `sharpness`: score relativo de gradiente (`SharpnessMeter`).
* `lat, lon, alt_m, h_acc_m, v_acc_m`: última leitura GNSS válida associada ao timestamp (vazio se indisponível).

---

## 3. `gnss.csv`

```csv
timestamp_s,lat,lon,alt_ellipsoidal_m,h_acc_m,v_acc_m,speed_mps,course_deg,fix_quality,source
```

* `fix_quality`: `0` = inválido, `1` = GPS autônomo (celular), `4` = RTK Fix, `5` = RTK Float.

---

## 4. `gcps.json`

Armazena pontos estacionários marcados pelo operador no campo (vértices do terreno, estacas do platô, extremidades de trena ou marcadores físicos):

```json
{
  "points": [
    {
      "id": "ESTACA-01",
      "kind": "stationary_gnss_avg",
      "timestamp_s": 1763217050.2,
      "sample_count": 30,
      "lat": -23.5505201,
      "lon": -46.6333094,
      "alt_ellipsoidal_m": 762.4,
      "h_acc_mean_m": 2.8,
      "v_acc_mean_m": 3.5,
      "arkit_xyz": [1.245, -0.312, -2.840],
      "frame_id": "000018",
      "notes": "Canto NE da area do galpao"
    }
  ],
  "scale_bars": [
    {
      "id": "TRENA-01",
      "point_a_id": "ESTACA-01",
      "point_b_id": "ESTACA-02",
      "measured_distance_m": 20.00
    }
  ]
}
```
