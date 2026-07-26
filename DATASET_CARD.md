# GIDD Dataset Card

## Intended use

GIDD supports research on image-based detection of glass-insulator damage in UAV inspection scenes. The detector predicts the categories and image-plane bounding boxes of insulators and visible damage regions. The dataset is not designed for estimating physical crack dimensions, damage severity, or remaining service life.

## Collection

- Collection period: September 2023 to June 2025
- Platforms: DJI Air 2 and DJI Mavic 3 UAVs
- Raw images: 2,753
- Retained images after manual screening: 2,542
- Scene factors include vegetation, transmission towers, rivers, buildings, and strong backlighting

Severely blurred, heavily occluded, and overexposed images were removed during data cleaning.

## Classes

- `insulator`: normal glass-insulator instances
- `defect`: cracks, self-explosion, and localized surface contamination

## Split

Before augmentation, the fixed 7:2:1 split contains:

| Partition | Images |
|---|---:|
| Training | 1,779 |
| Validation | 508 |
| Test | 255 |

The ratio was selected after preliminary comparisons of 6:2:2, 7:2:1, and 8:1:1 alternatives. The test set was held out and was not augmented. Offline augmentation increased the training and validation partitions to 8,895 and 2,540 images, respectively.

## Known limitations

- The current records do not establish whether adjacent UAV frames were grouped by flight, line, tower, or video sequence before splitting.
- Condition-stratified metrics for illumination, weather, occlusion, and target size are not yet available.
- Extreme backlighting, severe occlusion, very small damage regions, and crack-like reflections remain difficult conditions.
- The complete field dataset is subject to institutional review and cannot be uploaded before de-identification approval.

## De-identification requirements

Before release, remove or verify the absence of:

- EXIF and GPS metadata
- power-line, site, and tower identifiers
- faces, vehicle plates, and personal information
- internal filenames and directory structures
- confidential information in labels, logs, and configuration files
