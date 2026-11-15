# lidup

Plataforma aberta para levantamento topográfico rural, captura LiDAR/ARKit no iPhone e planejamento de terraplenagem/construções em terrenos rurais.

## Estrutura do Repositório

* [`docs/bundle_spec.md`](docs/bundle_spec.md) — Especificação única `lidup_bundle_v1` entre o iPhone e o PC.
* [`docs/ios_architecture_study.md`](docs/ios_architecture_study.md) — Análise técnica dos projetos open-source de referência.
* [`ios/LidupScanner/`](ios/LidupScanner/README.md) — App nativo SwiftUI (ARKit + LiDAR + alta resolução + filtro de nitidez + GNSS/Estacas) compilado para `.ipa` sem assinatura via [GitHub Actions](.github/workflows/ios-build.yml) para instalação com Sideloadly.
* [`processor/`](processor/README.md) — Motor Python (Windows/Linux) para ingestão, georreferenciamento SIRGAS 2000/UTM, classificação de solo, geração de `.laz`, MDT (`.tif`), curvas de nível (`.geojson`), malha `.obj` para Blender e simulador de corte/aterro.
