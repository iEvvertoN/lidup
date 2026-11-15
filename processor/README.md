# lidup-processor

Processador Python para transformar pacotes [`lidup_bundle_v1`](../docs/bundle_spec.md) capturados no iPhone em entregáveis topográficos e de terraplenagem para **QGIS**, **Blender** e **FreeCAD**.

## Instalação (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./processor[dev]"
```

## Uso rápido

1. Gerar um pacote sintético de terreno rural para validar o fluxo:
   ```powershell
   .\.venv\Scripts\lidup.exe make-sample --out data/sample_bundle
   ```
2. Processar o pacote (nuvem `.laz`, `mdt.tif`, `slope_pct.tif`, `contours.geojson`, `terrain_local.obj` e `report.json`):
   ```powershell
   .\.venv\Scripts\lidup.exe process data/sample_bundle --out outputs/sample
   ```
3. Simular o nivelamento de um platô para construção (ex.: galpão de 10×6 m centrado na origem local) calculando a cota de equilíbrio (`corte == aterro`):
   ```powershell
   .\.venv\Scripts\lidup.exe level outputs/sample/mdt.tif --out outputs/sample_leveled --local --min-x -5 --min-y -3 --max-x 5 --max-y 3
   ```
