# Internal Release Checklist

## Data privacy

- [ ] Remove EXIF and GPS metadata from every released image.
- [ ] Remove line, site, tower, and work-order identifiers.
- [ ] Check images for faces, plates, badges, screens, and documents.
- [ ] Replace internal filenames with neutral identifiers.
- [ ] Confirm institutional approval for the selected subset.

## Dataset integrity

- [ ] Add de-identified images and annotations.
- [ ] Add fixed `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`.
- [ ] Confirm whether adjacent UAV frames are grouped by sequence or asset.
- [ ] Add class-instance and target-size statistics.
- [ ] Verify that the 255-image test set has no augmented derivatives.
- [ ] Document the SFID partition and source version.

## Reproducibility

- [ ] Add the exact RFGAC-YOLO model configuration.
- [ ] Add training, validation, inference, and FPS scripts.
- [ ] Remove absolute local paths and credentials.
- [ ] Add package versions and installation instructions.
- [ ] Reproduce at least one table row from a clean environment.
- [ ] Document checkpoint provenance and license.

## Publication

- [ ] Choose and add a software/data license.
- [ ] Make the repository public.
- [ ] Create a versioned release tied to a fixed commit.
- [ ] Record the public repository URL, release URL, tag, and commit SHA.
- [ ] Update manuscript Data availability and Code availability statements.
