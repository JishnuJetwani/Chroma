# Original JPEG benchmark

The original JPEG model reached **23.98 dB PSNR / 0.8958 SSIM** on 824 held-out Expert C photos. This checkpoint produced the flower and Paris examples in the README. The [full FiveK run](full-fivek.md) expands this experiment to all 5,000 scenes.

## Protocol

- Custom split: 1,924 training / 824 test scenes; 1,958 / 838 editor pairs.
- Fixed curriculum: 10 global, 5 repair, and 10 joint epochs, starting from fresh weights.
- One H100 80GB, FP32, batch eight, accumulation one.
- Evaluation: valid content on padded 256-pixel images; final scheduled checkpoints.

## Expert C results

| Output | PSNR ↑ | SSIM ↑ | CIE76 color error ↓ |
|---|---:|---:|---:|
| Input | 18.339 dB | 0.7906 | 16.297 |
| Global | 21.974 dB | 0.8676 | 10.254 |
| Joint | 23.981 dB | 0.8958 | 8.346 |

Joint training adds **2.007 dB** over the global stage, with a paired scene-bootstrap 95% interval of **[1.748, 2.255]**. The joint checkpoint includes additional repair and joint training updates. Newly clipped highlight pixels increase by 0.227 percentage points [0.181, 0.281].

Editors A/B each have 17 training and seven test scenes. The main reported metrics use Expert C. The custom split and evaluation resolution define this benchmark; published comparisons require matching those settings.

## Runtime

Recorded GPU function time across attempts was **6.07 minutes**, including benchmarking, cache loading, training, checkpoint I/O, and evaluation. Estimated function compute was **$0.432**, excluding CPU preparation, startup/idle time, storage, and account adjustments.

[Metrics](jpeg-metrics.json) · [Model card](../model-card.md) · [Training workflow](../modal_training.md) · [Split registry](../../configs/splits/cloud_full_7030.json)
