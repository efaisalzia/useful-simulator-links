# Foundation Models for Remote Sensing and Smart Agriculture
## A Survey of Architectural Paradigms and Applications

---

## Slide 1 — Title

# Foundation Models for Remote Sensing and Smart Agriculture

**Subtitle:** Architectural Paradigms, Geospatial Intelligence, and Multimodal Convergence

**Author:** [Author Name]  
**Institution:** [Institution Name]  
**Date:** 2025

**Keywords:** Foundation Models · Remote Sensing · Smart Agriculture · Vision Transformers · SAM · Geospatial Intelligence · Multimodal Convergence

---

## Slide 2 — Introduction & Motivation

### The Paradigm Shift in Remote Sensing AI

**Before Foundation Models:**
- Task-specific architectures trained from scratch
- Limited generalisation across sensors and tasks
- High labelling cost for each new application

**After Foundation Models:**
- Scalable, universal representations via billion-parameter pretraining
- Single backbone → multiple downstream tasks
- Dramatic reduction in labelled data requirements

**Key Challenges Being Addressed:**
- High-resolution satellite imagery (sub-meter to 30 m)
- Multi-spectral and hyperspectral data fusion
- Temporal dynamics and change detection
- Geographic scale and distribution shift

**Why Now?**
> "Foundation models have initiated a paradigm shift moving beyond task-specific architectures toward scalable, universal representations." — Hong et al., 2025

*References: [hong2025foundation]*

---

## Slide 3 — Three Architectural Paradigms Overview

### Visual Diagram: Three Foundation Model Paradigms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   THREE ARCHITECTURAL PARADIGMS                         │
├──────────────────────┬────────────────────────┬────────────────────────┤
│   (a) ViT + MAE      │  (b) Hierarchical      │  (c) SAM-based         │
│                      │                        │                        │
│  Input               │  Input                 │   T1 ──┐               │
│    │                 │    │                   │        ▼               │
│  Masking (75%)       │  Stage 1 (H/4)         │  Frozen SAM Encoder    │
│    │                 │    │                   │        │               │
│  ViT Encoder         │  Stage 2 (H/8)         │  Adapters (trainable)  │
│    │                 │    │                   │        │               │
│  Latent Features     │  Stage 3 (H/16)        │   T2 ──┘               │
│    │                 │    │                   │        ▼               │
│  Lightweight         │  Stage 4 (H/32)        │  Feature Fusion        │
│  Decoder             │    │                   │        │               │
│    │                 │  Multi-scale           │  Change Map Output     │
│  Reconstruction      │  Feature Pyramid       │                        │
│                      │                        │                        │
│  SatMAE, Scale-MAE   │  RingMo, Swin-RS       │  SAM-CD, SCD-SAM      │
└──────────────────────┴────────────────────────┴────────────────────────┘
```

**Key Design Principle per Paradigm:**

| Paradigm | Core Idea | Spatial Resolution Handling |
|---|---|---|
| ViT + MAE | Global self-attention + masked reconstruction | Fixed patch size |
| Hierarchical (Swin) | Shifted-window attention + patch merging | Multi-scale pyramid |
| SAM-based | Frozen universal segmentor + domain adapters | Prompt-driven |

*References: [hong2025foundation]*

---

## Slide 4 — Vision Transformer + MAE

### ViT Architecture for Remote Sensing

**Core Characteristics:**
- Global self-attention across all image patches
- Masked Autoencoder (MAE) pretraining: 75% of patches masked
- Encoder processes only visible patches (25%) → efficient pretraining
- Lightweight decoder reconstructs masked regions

**Remote Sensing Adaptations:**
- **SatMAE**: Temporal + spectral positional encodings for multi-temporal imagery
- **Scale-MAE**: Scale-aware positional encodings to handle variable GSD
- **SpectralGPT**: Spectral-token design for hyperspectral data (128+ bands)

**Performance on EuroSAT Benchmark:**

| Model | Accuracy (%) | Parameters | Key Innovation |
|---|---|---|---|
| SpectralGPT | **99.21** | 632 M | Spectral tokenisation |
| SatMAE | 98.98 | 307 M | Temporal grouping |
| Scale-MAE | 98.74 | 307 M | Scale-aware PE |
| ViT-L (baseline) | 97.80 | 307 M | Standard MAE |

**Pretraining Data:** fMoW-Temporal (1M+ images), HyperionEO-1 (hyperspectral)

*References: [hong2024spectralgpt], [cong2022satmae]*

---

## Slide 5 — Hierarchical Transformers (Swin-based)

### Multi-Scale Feature Extraction for Dense Prediction

**Why Hierarchical?**
- Pixel-level tasks (segmentation, detection) need multi-scale features
- Global ViT produces single-resolution feature map → inadequate for dense tasks
- Shifted-window attention reduces complexity from O(N²) to O(N)

**Architecture: Swin Transformer for Remote Sensing**

```
Input (H×W) → Stage 1 (H/4, C) → Stage 2 (H/8, 2C)
            → Stage 3 (H/16, 4C) → Stage 4 (H/32, 8C)
            → FPN / UPerNet Head → Dense Prediction Output
```

**RingMo: Hierarchical Foundation Model**
- Pretrained on 2M+ remote sensing images
- Ring-shaped masked image modelling strategy
- Supports: scene classification, object detection, change detection, semantic segmentation

**Efficiency Results:**

| Configuration | Parameters | Accuracy | vs. Full Model |
|---|---|---|---|
| RingMo-Lite | 28 M | 85.3% | −14% params, −1.8% acc |
| RingMo-Base | 86 M | 87.2% | baseline | 
| RingMo-Large | 198 M | 88.1% | +130% params, +0.9% acc |
| **Pruned (40%)** | **~52 M** | **85.7%** | **>60% reduction, <2% loss** |

**Key Finding:** >60% parameter reduction achievable with <2% accuracy loss through structured pruning.

*References: [sun2022ringmo], [wang2025challenges]*

---

## Slide 6 — SAM-based Architectures

### Segment Anything Model for Remote Sensing Change Detection

**SAM-CD: Key Innovations**

1. **Dual-branch frozen encoder** — shared SAM image encoder for T1/T2 image pairs
2. **Trainable adapters** — lightweight domain adaptation without full fine-tuning
3. **Change-aware feature fusion** — bitemporal difference and attention-weighted fusion
4. **UNet-style decoder** — multi-scale skip connections for precise boundary recovery

**SAM-CD Architecture Flow:**
```
T1 Image ─► SAM Encoder (frozen) ─► Adapter ─►┐
                                               ├─► Fusion ─► UNet Decoder ─► Change Map
T2 Image ─► SAM Encoder (frozen) ─► Adapter ─►┘
              ▲_________Shared Weights_________▲
```

**Results on LEVIR-CD Benchmark:**

| Metric | SAM-CD | FC-EF | BIT | ChangeFormer |
|---|---|---|---|---|
| OA (%) | **99.14** | 98.72 | 98.75 | 98.65 |
| F1 (%) | **95.50** | 83.40 | 89.31 | 90.40 |
| mIoU (%) | **91.68** | 71.46 | 80.68 | 82.48 |

**Sample Efficiency (LEVIR-CD):**

| Training Data | mIoU (%) |
|---|---|
| 10% | 76.42 |
| 20% | 80.15 |
| 40% | **83.17** |
| 100% | 91.68 |

> With only 40% of training data, SAM-CD achieves 83.17% IoU — demonstrating strong sample efficiency.

*References: [ding2024samcd]*

---

## Slide 7 — SAM Limitations

### When SAM Falls Short in Remote Sensing

**Five Core Limitations Identified:**

```
┌──────────────────────────────────────────────────────────┐
│                    SAM LIMITATIONS                       │
├──────────────────────────────────────────────────────────┤
│  1. INTERACTIVE NATURE                                   │
│     Requires user prompts (points/boxes/masks)           │
│     → Not suitable for fully automated batch processing  │
├──────────────────────────────────────────────────────────┤
│  2. LOW-CONTRAST SENSITIVITY                             │
│     Struggles with spectrally similar land cover types   │
│     → Agricultural fields, tundra, desert surfaces       │
├──────────────────────────────────────────────────────────┤
│  3. SMALL OBJECT HANDLING                                │
│     Objects < 16×16 pixels frequently missed or merged   │
│     → Vehicles, buildings in low-resolution imagery      │
├──────────────────────────────────────────────────────────┤
│  4. RESOLUTION SENSITIVITY                               │
│     Pretrained on natural images (typically ≤1024×1024)  │
│     → Performance degrades on very-high-res RS imagery   │
├──────────────────────────────────────────────────────────┤
│  5. LINEAR FEATURE SEGMENTATION                          │
│     Roads, rivers, field boundaries poorly delineated    │
│     → Thin elongated structures missed or fragmented     │
└──────────────────────────────────────────────────────────┘
```

**Mitigation Strategies:**
- Adapters and fine-tuning on RS-specific data (SAM-CD approach)
- Multi-scale sliding window inference for high-resolution images
- Automatic prompt generation using domain-specific detectors
- Hybrid approach: SAM for region proposals + RS model for classification

*References: [huo2025when]*

---

## Slide 8 — Self-Supervised Learning Strategies

### Pretraining Without Labels

**Three SSL Paradigms in Remote Sensing:**

| Strategy | Mechanism | Representative Models | Best For |
|---|---|---|---|
| **MAE** | Reconstruct masked patches | SatMAE, Scale-MAE, SpectralGPT | Scene classification, segmentation |
| **Contrastive** | Pull positives together, push negatives apart | MoCo-RS, SimCLR-RS | Cross-sensor, cross-time |
| **Hybrid** | MAE + contrastive objectives | SkySense, SaCo | Multi-task, multi-modal |

**SkySense: State-of-the-Art Hybrid SSL**

Architecture:
- Multi-granularity contrastive learning (pixel, region, image level)
- Temporal MAE branch for multi-date imagery
- 2.4B parameters pretrained on 21.5M RS images (Gaofen, Sentinel, Landsat)

**SkySense Benchmark Results:**

| Task | Dataset | Metric | SkySense | Previous SoTA |
|---|---|---|---|---|
| Change Detection | LEVIR-CD | F1 (%) | **92.58** | 90.40 |
| Semantic Seg. | iSAID | mIoU (%) | **70.91** | 68.34 |
| Scene Classification | AID | Acc (%) | **97.20** | 96.84 |
| Object Detection | DOTA-v1.0 | mAP (%) | **81.43** | 79.82 |

**SaCo (Spatial-Contextual Contrastive):** Leverages geographic co-occurrence for positive pair mining — nearby geolocations share semantic context.

*References: [guo2024skysense], [stival2025saco]*

---

## Slide 9 — Domain Adaptation

### Generic vs. Domain-Specific Pretraining

**The Core Question:**
> When does remote sensing domain-specific pretraining outperform ImageNet-pretrained models?

**Factors Favouring Domain-Specific Pretraining:**
- Target task involves multi-spectral data (> 3 bands)
- Fine-grained agricultural or geological classes
- Limited labelled data available for target task
- Significant domain gap (aerial vs. natural scene textures)

**Factors Favouring Generic (ImageNet) Pretraining:**
- Large labelled target dataset available (> 50K images)
- Task semantics align well with natural images
- Computational budget is constrained

**GeRSP: Generalised Remote Sensing Pretraining**

| Pretraining | UC Merced | AID | NWPU-RESISC45 | ISPRS Potsdam |
|---|---|---|---|---|
| ImageNet ViT-L | 96.7% | 94.2% | 92.8% | 83.1% |
| SatMAE | 97.1% | 95.3% | 93.6% | 85.4% |
| **GeRSP** | **98.4%** | **96.8%** | **95.1%** | **87.9%** |

**Brain-Inspired Approach (BraIn):** Integrates cognitive-inspired feature hierarchies to improve generalisation across sensor modalities.

**Key Insight:** Domain gap is most pronounced for hyperspectral and SAR data; RGB-only tasks show smaller benefits from RS-specific pretraining.

*References: [huang2024gersp], [jiao2023brain]*

---

## Slide 10 — Prithvi: NASA-IBM Foundation Model

### Earth Observation at Planetary Scale

**Model Overview:**
- Developed jointly by NASA and IBM Research
- Architecture: Temporal ViT with multi-spectral patch embeddings
- Pretraining: >1 TB of Harmonized Landsat-Sentinel (HLS) data
- Spectral bands: 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
- Temporal context: 3-frame temporal sequences

**Performance on Key Benchmarks:**

| Task | Dataset | Metric | Prithvi-100M | Prithvi-300M | Supervised Baseline |
|---|---|---|---|---|---|
| Flood Mapping | Sen1Floods11 | IoU (%) | 83.4 | **86.2** | 79.1 |
| Crop Segmentation | CropHarvest | F1 (%) | 79.3 | **82.7** | 74.5 |
| Wildfire Scar | NASA HLS | IoU (%) | 74.8 | **78.1** | 68.3 |
| Multi-temporal Seg. | TimeSen2Crop | Acc (%) | 88.6 | **91.3** | 85.2 |

**Sample Efficiency:**

| Labelled Data | Prithvi-300M | Supervised (ResNet-50) |
|---|---|---|
| 1% | 61.3% IoU | 38.7% IoU |
| 10% | 74.5% IoU | 62.1% IoU |
| 100% | 86.2% IoU | 79.1% IoU |

**Trade-offs:**
- ✅ Exceptional for time-series agricultural applications
- ✅ Open-weight model available on HuggingFace
- ⚠️ Limited to Landsat/Sentinel spatial resolution (10–30 m)
- ⚠️ SAR and thermal infrared not included in v1

*References: [jakubik2023prithvi], [hsu2025prithvi]*

---

## Slide 11 — Vision-Language Models

### Bridging Satellite Observation and Language

**Why Vision-Language for Remote Sensing?**
- Natural language queries for geospatial analysis
- Zero-shot classification without task-specific labels
- Grounded captioning and visual question answering
- Connecting satellite imagery to textual reports/metadata

**Key Models:**

| Model | Training Data | Architecture | Key Capability |
|---|---|---|---|
| **RemoteCLIP** | 12 RS datasets + web text | CLIP ViT-L/14 | RS-specific image-text alignment |
| **RS5M** | 5M RS image-caption pairs | CLIP + expansion | Large-scale RS pretraining |
| **GeoRSCLIP** | 10M RS pairs (YFCC geotag) | CLIP ViT-H/14 | Geo-aware representation |
| **RSGPT** | Instruction-tuned RS pairs | LLaVA-style | RS visual question answering |

**RemoteCLIP Zero-Shot Results (vs. CLIP ViT-L/14):**

| Dataset | CLIP (%) | RemoteCLIP (%) | Improvement |
|---|---|---|---|
| UCM-Captions | 64.2 | **81.3** | +17.1 |
| RSICD | 58.7 | **74.9** | +16.2 |
| RSITMD | 61.5 | **79.4** | +17.9 |

**Prompt Engineering Matters:**
```
Weak prompt:   "a satellite image"        → 64.2% accuracy
Medium prompt: "aerial view of {class}"   → 74.8% accuracy  
Strong prompt: "a satellite image of {class} captured from above, 
                showing {context}"        → 81.3% accuracy
```

*References: [liu2024remoteclip], [zhang2024rs5m]*

---

## Slide 12 — Smart Agriculture Applications

### Foundation Models in the Field

**Four Primary Application Domains:**

**1. Crop Type Segmentation**
- Multi-spectral time-series segmentation of field boundaries
- Prithvi achieves 82.7% F1 on CropHarvest with 10× less labelled data
- Supports 40+ crop classes across continental scales

**2. Change Detection for Agriculture**
- Monitoring deforestation, irrigation expansion, field conversion
- SAM-CD: 95.50% F1 on LEVIR-CD benchmark
- Multi-date Sentinel-2 pairs with 5-day revisit cycle

**3. Multi-Temporal Crop Analysis**
- Phenological stage detection (planting → emergence → harvest)
- TimeSen2Crop dataset: 10 temporal observations per season
- Prithvi temporal attention captures crop growth trajectory

**4. Cost-Effectiveness Findings (Sosa et al., 2024):**

| Approach | Labelled Data Required | mIoU | Annotation Cost |
|---|---|---|---|
| Train from scratch | 10,000 fields | 71.3% | $$$$ |
| ImageNet fine-tuning | 5,000 fields | 78.6% | $$$ |
| **Foundation model (Prithvi)** | **500 fields** | **79.8%** | **$** |
| Foundation model (100%) | 10,000 fields | 84.2% | $$$$ |

> **Key finding:** Foundation models achieve comparable performance with **20× less labelled data**, reducing annotation costs by up to 90%.

*References: [sosa2024effective]*

---

## Slide 13 — Open Challenges

### Barriers to Deployment

**Data Challenges:**

```
┌─────────────────────────────────────────────────────┐
│  DATA CHALLENGES                                    │
│                                                     │
│  Scale Gap                                          │
│  • Natural image datasets: billions of images       │
│  • RS labelled datasets: typically <100K images     │
│  • Consequence: pretrain-finetune gap remains large │
│                                                     │
│  Geographic Bias                                    │
│  • Most RS datasets: North America + Europe         │
│  • Global South under-represented                   │
│  • Models fail on tropical agriculture, arid zones  │
│                                                     │
│  Temporal Consistency                               │
│  • Sensor changes, orbit drift, seasonal variation  │
│  • Domain shift across acquisition dates            │
└─────────────────────────────────────────────────────┘
```

**Architectural Challenges:**

```
┌─────────────────────────────────────────────────────┐
│  ARCHITECTURAL CHALLENGES                           │
│                                                     │
│  Computational Demands                              │
│  • SkySense (2.4B params): needs 64× A100 GPUs      │
│  • Prithvi-300M: 6 TB disk for pretraining data     │
│  • Inference: >8 GB VRAM for full-resolution RS     │
│                                                     │
│  Edge Deployment                                    │
│  • UAV/satellite on-board processing limited        │
│  • Quantisation and pruning reduce accuracy         │
│  • Real-time crop monitoring needs <100ms latency   │
│                                                     │
│  Multi-Modal Integration                            │
│  • SAR + optical + hyperspectral fusion unsolved    │
│  • Cross-sensor pretraining alignment               │
└─────────────────────────────────────────────────────┘
```

*References: [xiao2025foundation], [sang2026onboard]*

---

## Slide 14 — Future Directions

### The Road Ahead

**1. Hybrid SSL Frameworks**
- Combine MAE reconstruction + contrastive objectives + temporal consistency
- Leverage geographic metadata as self-supervised signal
- Cross-modal pretraining (optical ↔ SAR ↔ hyperspectral)

**2. Efficiency and Edge Deployment**
- Knowledge distillation from billion-parameter teachers → edge models
- Hardware-aware NAS for UAV-deployable foundation models
- ORBIT: On-board real-time inference for satellite platforms (Bai et al., 2025)

**3. Multimodal Convergence**
- Unified vision-language-action models for geospatial AI
- Natural language control of change detection and crop monitoring
- RS-specific LLMs grounded in satellite observation chains

**4. Agro-Intelligence Systems**

```
Satellite Imagery
      │
      ▼
Foundation Model Encoder ──► Temporal Features
      │                            │
      ▼                            ▼
Language Grounding ◄────── Crop State Estimation
      │
      ▼
Actionable Recommendations
("Field 42B shows 15% NDVI decline — 
  irrigation recommended within 72h")
```

**5. Federated and Privacy-Preserving Learning**
- Collaborative pretraining across national boundaries
- Sensitive agricultural data privacy preservation

*References: [hong2025foundation], [bai2025orbit]*

---

## Slide 15 — Conclusion

### Key Takeaways

**Architectural Landscape:**
- Three paradigms dominate: ViT+MAE, Hierarchical (Swin), SAM-based
- Each paradigm offers distinct trade-offs in resolution, efficiency, and task suitability

**Performance Highlights:**

| Model | Task | Best Result |
|---|---|---|
| SpectralGPT | Scene classification (EuroSAT) | 99.21% accuracy |
| SAM-CD | Change detection (LEVIR-CD) | F1: 95.50%, mIoU: 91.68% |
| SkySense | Multi-task RS | F1: 92.58%, mIoU: 70.91% |
| Prithvi-300M | Flood mapping | 86.2% IoU |
| RemoteCLIP | Zero-shot classification | +17% over CLIP baseline |

**Sample Efficiency Revolution:**
- Foundation models achieve equivalent performance with **10–20× less labelled data**
- Critical for agriculture where ground-truth collection is expensive

**Path Forward:**
- Billion-scale pretraining has advanced RS perception capabilities significantly
- Convergence of satellite perception + vision-language reasoning is imminent
- **Integrated agro-intelligence systems** — bridging observation with actionable, language-grounded understanding — represent the next frontier

> "The architectural groundwork surveyed here serves as the essential prerequisite for the convergence of satellite-scale perception with vision-language reasoning."

---

## Slide 16 — References

### Citations

| Key | Reference |
|---|---|
| [bai2025orbit] | Bai, J. et al. (2025). ORBIT: On-Board Real-Time Inference for Satellite Foundation Models. *arXiv preprint*. |
| [cong2022satmae] | Cong, Y. et al. (2022). SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. *NeurIPS 2022*. |
| [ding2024samcd] | Ding, L. et al. (2024). SAM-CD: Change Detection with Segment Anything Model. *IEEE TGRS*. |
| [guo2024skysense] | Guo, X. et al. (2024). SkySense: A Multi-Modal Remote Sensing Foundation Model. *CVPR 2024*. |
| [hong2024spectralgpt] | Hong, D. et al. (2024). SpectralGPT: Spectral Foundation Model. *IEEE TPAMI*. |
| [hong2025foundation] | Hong, D. et al. (2025). Foundation Models for Remote Sensing: A Survey. *IEEE TGRS*. |
| [hsu2025prithvi] | Hsu, C. et al. (2025). Prithvi-EO 2.0: Scaling NASA-IBM Earth Observation Foundation Model. *arXiv*. |
| [huang2024gersp] | Huang, Z. et al. (2024). GeRSP: Generalised Remote Sensing Pretraining. *IGARSS 2024*. |
| [huo2025when] | Huo, C. et al. (2025). When SAM Meets Remote Sensing: Limitations and Adaptations. *IEEE GRSL*. |
| [jakubik2023prithvi] | Jakubik, J. et al. (2023). Foundation Models for Geospatial Data — Prithvi. *arXiv:2310.18660*. |
| [jiao2023brain] | Jiao, L. et al. (2023). Brain-Inspired Remote Sensing Foundation Model. *IEEE TGRS*. |
| [liu2024remoteclip] | Liu, F. et al. (2024). RemoteCLIP: A Vision-Language Foundation Model for Remote Sensing. *IEEE TGRS*. |
| [sang2026onboard] | Sang, N. et al. (2026). On-Board Deployment of Remote Sensing Foundation Models. *IEEE JSTARS*. |
| [sosa2024effective] | Sosa, M. et al. (2024). Effective Use of Foundation Models in Agricultural Remote Sensing. *Remote Sensing*. |
| [stival2025saco] | Stival, L. et al. (2025). SaCo: Spatial-Contextual Contrastive Learning for Remote Sensing. *IGARSS 2025*. |
| [sun2022ringmo] | Sun, X. et al. (2022). RingMo: A Remote Sensing Foundation Model with Masked Image Modeling. *IEEE TGRS*. |
| [wang2025challenges] | Wang, W. et al. (2025). Challenges and Advances in Remote Sensing Foundation Models. *IEEE GRSM*. |
| [xiao2025foundation] | Xiao, Z. et al. (2025). Foundation Models for Earth Observation: Open Challenges. *Nature Reviews*. |
| [zhang2024rs5m] | Zhang, Z. et al. (2024). RS5M and GeoRSCLIP: A Large Scale Vision-Language Dataset for Remote Sensing. *IEEE TGRS*. |

---

## Slide 17 — Thank You

# Thank You

## Questions & Discussion

---

**Contact:** [author@institution.edu]  
**Preprint / Paper:** [DOI or arXiv link]  
**Model Weights:** [HuggingFace / GitHub URL]

---

### Discussion Prompts

1. How should geographic bias in RS pretraining datasets be addressed for global agricultural applications?

2. What is the most promising path to real-time foundation model inference on edge devices (UAV/satellite)?

3. Can a single unified foundation model replace task-specific architectures across all RS modalities?

4. How do we handle the temporal dynamics of agricultural systems within current ViT-based frameworks?

---

*Slides available at: [URL]*  
*This work supported by: [Funding Agency]*
