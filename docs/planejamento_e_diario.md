# `lidup` — Planejamento dos Próximos Passos e Diário de Bordo

> Documento vivo para acompanhar a evolução científica e prática do projeto (`LidupScanner` iOS + processador Python `lidup`), garantindo testes **Antes vs Depois** totalmente reversíveis.

---

## 1. Regra de Ouro de Segurança e Reversibilidade

Antes de qualquer alteração nos algoritmos:

1. **Capturas brutas são imutáveis:** nenhuma etapa do processador altera as pastas de captura (`YYYY-MM-DD_HHMMSS_*`).
2. **Baseline versionado no Git:** o estado estável de referência está salvo no commit `f4204d3`.
3. **Comparação A/B obrigatória via CLI:** toda nova técnica entra primeiro como **opção/flag** mantendo o método antigo intacto, para gerar `out_antes` e `out_depois` sobre o **mesmo pacote bruto** e comparar no `lidup view` e no `report.json`.
4. **Sem necessidade de recompilar o iPhone agora:** as melhorias das Frentes 1 e 2 rodam 100% no PC (`processor/src/lidup/`).

---

## 2. Planejamento dos Próximos Passos (Priorizado por Retorno Real / ROI)

### Etapa 0 — Registrar o Baseline ("Antes") do Objeto 360º
**Objetivo:** congelar a referência exata com o algoritmo atual (`f4204d3`) antes de aplicar qualquer mudança.

- [ ] Copiar a nova pasta de captura 360º para a raiz do projeto (ou `data/`).
- [ ] Rodar o pipeline atual sem modificações:
  ```powershell
  .\.venv\Scripts\lidup.exe process <PASTA_OBJETO_360> --out out_360_antes --voxel-size 0.02 --refine-poses
  .\.venv\Scripts\lidup.exe view out_360_antes\cloud_local.ply
  ```
- [ ] Anotar na tabela de benchmark (Seção 4) o total de pontos, qualidade visual no fechamento do giro 360º (presença de "casca dupla", degrau no encontro inicial/final, ruído de fundo) e tempo de processamento.

---

### Frente 1 — Topografia e Drenagem Rural (Maior ROI para 5–7 ha)

| Ordem | Melhoria | Arquivo alvo | Por que traz retorno real | Critério de aceite (Antes vs Depois) |
| :--- | :--- | :--- | :--- | :--- |
| **1A** | **Travamento na Gravidade no Georreferenciamento** (ajuste horizontal Yaw + XY + $\Delta Z$, sem inclinar *pitch*/*roll* pelo GPS) | `processor/src/lidup/georef.py` | Hoje `umeyama_alignment` ajusta rotação 3D livre contra `alt_m` do GNSS (que oscila ~11 m no celular, ex.: `v_acc_m=11.266` em campo), podendo entortar o plano do terreno e distorcer `slope_pct.tif` e drenagem, desperdiçando a gravidade precisa da IMU/ARKit. | Preservar 100% a vertical da IMU; manter modo antigo via flag para comparação em capturas com GNSS. |
| **1B** | **Filtro Físico de Solo CSF (*Cloth Simulation Filter*)** como opção A/B (`--ground-method csf` vs `percentile`) | `processor/src/lidup/terrain.py` | Em pasto com capim alto, touceiras e taludes, a grade atual (`cell_size_m=1.0`, percentil 5 + `median_filter`) tende a englobar base de vegetação ou suavizar cristas; o CSF simula um tecido invertido sob gravidade e preserva melhor quebras de relevo. | Comparar `point_count_ground`, curvas de nível e `terrain_local.obj` em captura rural real. |

---

### Frente 2 — Objetos 3D, Estruturas e Interiores (Validação imediata no Objeto 360º)

| Ordem | Melhoria | Arquivo alvo | Por que traz retorno real (e limites) | Critério de aceite (Antes vs Depois) |
| :--- | :--- | :--- | :--- | :--- |
| **2A** | **Filtro / Ponderação Física do LiDAR ($\frac{\cos\theta}{z^2}$)** | `processor/src/lidup/bundle.py` | Pela física do dToF SPAD, a variância cresce com $z^2$ e ângulos rasantes ($\theta$ alto) alongam o feixe e aumentam ruído/viés. Priorizar raios mais frontais e próximos reduz "borrão" em bordas e superfícies especulares. | Menos pontos fantasmas e bordas mais nítidas no `out_360_depois` sem perder cobertura útil. |
| **2B** | **Open3D `PoseGraph` (Loop Closure) + `Colored ICP` / `TSDF` restrito ao modo Objeto/Interno** | `processor/src/lidup/bundle.py` | O ICP sequencial atual ainda acumula deriva ao dar a volta completa (360º). Fechar o laço global via `PoseGraph` elimina o degrau onde o fim encontra o início; `Colored ICP` trava deslizamento em superfícies lisas. **Restrição importante:** `Colored ICP` falha sob sol/sombra variável ao ar livre e `TSDF` fino estoura RAM em 5–7 ha — usar **apenas** para objetos/interiores. | Redução visível de superfície dupla ("fantasma" ao completar 360º) no objeto de teste. |

---

### Frente 3 — Escala para 5–7 Hectares e Itens Baixo-ROI (Adiados de propósito)

- **Protocolo de Campo (alto impacto sem código):** caminhar em **circuitos fechados** (*loop closure* físico), marcar estacas (`gcps.json`) e medir ao menos uma distância de trena para mitigar deriva VIO ("efeito banana") em trajetos acima de 50–100 m.
- **Correção Geoidal ($H = h - N$, EGM2008 / MAPGEO):** **adiada** — em 5–7 ha (~250 m), o gradiente geoidal varia só alguns centímetros frente a ~10 m de erro vertical do GPS de celular, portanto não melhora drenagem local. Só entra quando houver uso com marcos IBGE ou receptor **RTK-GNSS** externo.
- **Regularização 3DGS estilo DN-Splatter:** **adiada** — depende de GPU dedicada e customização do treino `gsplat`; fazer só depois que poses e nuvem base estiverem travadas.

---

## 3. Diário de Bordo

### 2026-11-15 — Marcos Concluídos até o Momento

| Hora / Commit | Marco / Teste | O que foi observado | Ação tomada / Resultado |
| :--- | :--- | :--- | :--- |
| `3c1e645`, `625803a` | Fundação do projeto (`LidupScanner` + `lidup`) | Estrutura inicial iOS SwiftUI/ARKit + processador Python topográfico (`LAZ`, `MDT`, `slope_pct`, `contours`, `OBJ`, `level`). | Base funcional criada no repositório. |
| `d2c37b7` | Correção de build iOS no CI | XcodeGen sobrescrevia o `Info.plist` removendo chaves de permissão de câmera/localização. | Ajustados `project.yml`, `Resources/Info.plist` e `.github/workflows/ios-build.yml` (CI verde). |
| `dd0300b`, `0be96fd` | **Teste 1:** Mesa / computador interno (`2026-11-15_133839_galpao_norte`) | Faltava exportação de nuvem local simples, dataset 3DGS, visualizador direto no PC e feedback de malha no iPhone. | Adicionados `cloud_local.ply`, `export-3dgs`, `lidup view` (com atalhos `W/S/I/O/A/D/R`), wireframe `ARMeshAnchor` e preset `Objeto / 3DGS (0.25m)`. |
| `f4204d3` | **Teste 2:** Quarto, 170 MB, 41 frames (`2026-11-15_141709_galpao_norte`) | Movimento frente/trás revelou dessincronia entre pose da foto RGB e timestamp do `depthMap` (ex.: salto de `0.321 m` / `14.03°` entre frames 30→31) + viés de borda. | iOS passou a gravar `depth_tx..depth_qw` e usar `.gravity` no modo objeto; Python ganhou reprojeção RGB, filtro de salto de borda (`edge_jump`) e `--refine-poses` (ICP). Resultado `out_quarto_limpo`: `352.133` pts (`77.932` solo / `274.201` não-solo); `pytest` com 4/4 passando. |
| Pós-`f4204d3` | **Teste 3:** Lateral de carro (`2026-11-15_150226_galpao_norte` → `out_novo`) | Menos erros com captura pausada e poses separadas confirmadas no CSV (`tx` vs `depth_tx`), porém `frames.csv` mostrou GPS com `h_acc_m=2.466` e `v_acc_m=11.266`, e lataria brilhante apresentou ruído especular/rasante. | Pesquisa científica mapeou 4 alavancas reais e identificou o risco da rotação 3D livre em `georef.py` entortar terrenos pelo erro vertical do GPS. |
| `2026-11-15` (16:43) | **Teste 4:** Objeto 360º (`2026-11-15_164308_galpao_norte` → `out_360_antes`) | 23 frames (`0.25m`), `focus_locked: true`, GNSS ativo (`frame_gnss`, `EPSG:31984`, `rmse_total=1.304m`). **Descoberta crítica confirmada nos dados:** a rotação 3D livre (`umeyama_alignment`) inclinou a gravidade em **128,89°** (`up_vec[2] = -0.628`) por tentar ajustar um giro pequeno (~3 m) ao ruído vertical/horizontal do GPS de celular. | Baseline `out_360_antes` gerado (`91.918` pts totais / `23.047` solo) e visualizador aberto; comprova na prática a necessidade imediata da **Frente 1A** (travamento na gravidade). |

---

## 4. Tabela de Comparação Antes vs Depois (Benchmark Objeto 360º)

> Preencher após rodar a Etapa 0 e cada etapa das Frentes 1 e 2.

| Versão / Experimento | Comando / Flags usadas | Pontos totais | Pontos de solo | Fechamento 360º (degrau / casca dupla) | Ruído em bordas / superfícies lisas | Notas visuais (`lidup view`) |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Antes (Baseline `f4204d3`)** | `lidup process 2026-11-15_164308_galpao_norte --out out_360_antes --voxel-size 0.02 --refine-poses` | `91.918` | `23.047` | **Falhou (3 cópias deslocadas; saltos ICP até 0,886 m / 25,3°)** | Bordas rasantes mantidas | Tombado **128,89°** pelo ajuste 3D livre ao GNSS |
| **Depois 1A+1B+2A+2B (`gravity_locked` + `CSF` + $\cos\theta/z^2$ + `PoseGraph`)** | `lidup process 2026-11-15_164308_galpao_norte --out out_360_depois --voxel-size 0.02 --refine-poses` | `88.748` | `70.588` | **Alinhamento global consistente (RMSE 6 cm: 2,49 cm → 2,01 cm; fitness 0,563 → 0,639)** | Filtrado acima de `75°` + voxel ponderado | Gravidade **0,00°** de inclinação; chão/objeto corretamente separados (`18.160` pts no objeto) |
| **Depois TSDF + Cor 12 MP (`--tsdf --tsdf-voxel 0.008`)** | `lidup process 2026-11-15_164308_galpao_norte --out out_360_tsdf --tsdf --tsdf-voxel 0.008` | `170.988` vértices / pontos (`329.914` triângulos) | `137.757` | **Malha 3D contínua via `UniformTSDFVolume` + `PoseGraph` conectado** (`33.231` pts no objeto) | Textura projetada dos JPEGs `4032x3024` (com checagem de oclusão e normal) | Gravidade **0,00°** (`gravity_locked`); exportado em `out_360_tsdf/object_mesh_local.ply` |
| **Depois Super-Res 2x + Foco Órbita (`--depth-upsample 2 --tsdf-focus-radius 1.5 --tsdf-voxel 0.006`)** | `lidup process 2026-11-15_164308_galpao_norte --out out_360_super --tsdf --tsdf-voxel 0.006 --depth-upsample 2 --tsdf-focus-radius 1.5` | `226.909` vértices / pontos (`446.935` triângulos) | `193.344` | **Foco automático no centro de convergência 360° + super-resolução guiada 2x (`512x384`)** (`33.565` pts no objeto) | Unmixing bimodal de bordas RGB + projeção `12 MP` sem *flying pixels* | Volume concentrado (`3.0 m`, voxel `7.5 mm`, `+35,5%` triângulos) em `out_360_super/object_mesh_local.ply` |
| **Objeto próximo `213918` (TSDF 4 mm inicial)** | `lidup process 2026-11-15_213918_galpao_norte --out out_213918 --max-depth 2.5 --tsdf --tsdf-voxel 0.004 --depth-upsample 2 --tsdf-focus-radius 0.80` | `350.824` vértices (`694.552` triângulos) | `314.336` | PoseGraph grosso (`2.5 cm`) + `sdf_trunc = 30 mm` (`36.488` pts no objeto) | `22/54` frames sem upsample por rotação `> 0.5°` entre RGB e depth | Base antes da análise detalhada |
| **Objeto próximo `213918` (Foco 2 mm + Warp RGB + `sdf_trunc 7 mm`)** | `lidup process 2026-11-15_213918_galpao_norte --out out_213918 --max-depth 2.0 --tsdf --tsdf-voxel 0.002 --depth-upsample 3 --tsdf-focus-radius 0.48` | `661.653` vértices (`1.305.317` triângulos) | `494.052` | **Reprojeção RGB↔LiDAR (`54/54` `12 MP` em `768x576`) + `480³` (`2.0 mm`) + Taubin 2x (`167.601` pts no objeto)** | Cor bilinear subpixel ponderada por nitidez + oclusão `2.5 cm` | **`4,6x` mais vértices só no objeto** (`out_213918/object_mesh_local.ply`) |
| **Objeto próximo `213918` (`--mvs --photometric-opt --tsdf-voxel 0.001 --tsdf-focus-radius 0.24`)** | `lidup process 2026-11-15_213918_galpao_norte --out out_213918 --max-depth 1.6 --tsdf --tsdf-voxel 0.001 --depth-upsample 3 --tsdf-focus-radius 0.24 --mvs --photometric-opt` | `444.448` vértices (`875.858` triângulos) | `444.448` (`99.907` acima `z>0.01`) | **Filtro piso cerâmico + `LiDAR-Seeded MVS` (`±12 mm` ZNCC em `768x576`) + `ColorMapOptimizer` + peso `imu.csv` + voxel `1,00 mm`** | ZERO segundo chão (`z_min=-0.0486m`); cor `12 MP` otimizada fotometricamente | Órbita 360° limpa e texturas ajustadas diretamente pelas fotos (`out_213918/object_mesh_local.ply`) |
| **Quarto `085439` (TSDF sala inteira + Fotometria `12 MP`)** | `lidup process 2026-11-16_085439_galpao_norte --out out_085439 --max-depth 4.5 --tsdf --tsdf-voxel 0.008 --depth-upsample 2 --photometric-opt` | `289.524` vértices (`555.610` triângulos) | `28.543` (`260.981` paredes/teto/móveis) | **Reconstrução do cômodo (`4,58 × 2,33 × 3,07 m`) com super-resolução `2x` (`512x384`) + `ColorMapOptimizer`** | Modo `local_only` (GNSS interno `±6,6 m` isolado) + peso `imu.csv` | Malha contínua em `out_085439/object_mesh_local.ply` (`22` fotos `12 MP`) |
