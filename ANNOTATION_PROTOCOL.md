# Annotation Protocol

## Annotation targets

Each image is annotated for two object-detection classes:

- `insulator`: the visible glass-insulator assembly
- `defect`: visible crack, self-explosion, or localized surface contamination

Annotations represent image-plane bounding boxes. They do not encode physical crack dimensions or severity grades.

## Quality control

Two researchers independently annotated and cross-checked all retained samples. Disagreements were resolved through discussion.

## Items to finalize before public release

- Name and version of the annotation tool
- Bounding-box inclusion rules for partially occluded objects
- Rules for multiple nearby defect regions
- Minimum visible size accepted as a defect
- Treatment of ambiguous reflections and contamination
- Inter-annotator agreement or disagreement counts, if retained records permit
- Final label format and class-ID mapping
