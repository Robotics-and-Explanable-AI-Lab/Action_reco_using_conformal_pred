# Assembly101 TSM Baseline – Replication Instructions

This repository contains the code to train, validate, and generate test set predictions for fine‐grained action recognition on the Assembly101 dataset using a Temporal Shift Module (TSM) network.

The implementation is adapted from the original TSM repository and has been modified to work with the Assembly101 annotations and data splits.

---
## Overview
The Assembly101 dataset includes videos recorded from **12 synchronized camera views**:
- **8 static RGB views**
- **4 egocentric monochrome views**

This repository provides code for:
1. **Data Preparation**: Generating TSM-style annotations from the provided CSV files.
2. **Training**: Fine-tuning a TSM model (e.g., starting from an EPIC-KITCHENS pretrained model) on Assembly101.
3. **Evaluation**: Generating predictions on validation or test splits and saving the outputs as NumPy files (`preds.npy` and `scores.npy`).

---
## Requirements
Ensure that you have the following dependencies installed:

- Python 3.6+
- PyTorch (>=1.6)
- torchvision
- NumPy
- Other standard packages (`PIL`, `tqdm`, etc.)

**Recommended**: Use a virtual environment (e.g., `conda` or `venv`) to manage dependencies.

---
## Repository Structure
```
.
├── main.py                     # Training and validation script
├── test_models.py              # Evaluation script (loads a pretrained model and generates predictions)
├── gen_fine_labels.py          # Converts CSV annotations into TSM-style text files
├── ops/                        # Contains dataset classes and model definitions
├── pretrained_models/          # Pretrained TSM model checkpoints (download as needed)
├── data/                       # Contains TSM-style annotation files and category.txt (listing 1380 fine-grained actions)
```

---
## Data Preparation
### Annotation Generation
Convert CSV annotations into TSM-style text files using the provided script:
```bash
python gen_fine_labels.py combined test
```
Use `rgb` or `mono` instead of `combined` if working with a single modality.

### Directory Structure for Frames
Ensure your extracted frames follow the naming convention expected by the dataset class. Example format for RGB frames:
```
C10095_rgb_frame_000086.jpg
```
Modify `image_tmpl` in the dataset code if your naming convention is different.

### Data Splits
Place the generated annotation files in the `data/` folder:
- `train_combined.txt`
- `validation_combined.txt`
- `test_combined.txt`

---
## Training
To fine-tune the model on Assembly101, run:
```bash
python main.py Assembly101 combined \
  --arch resnet50 --num_segments 8 --gd 20 \
  --lr 0.001 --lr_steps 20 40 --epochs 50 \
  --batch-size 64 -j 16 --dropout 0.5 \
  --consensus_type=avg --eval-freq=1 \
  --shift --shift_div=8 --shift_place=blockres --npb \
  --tune_from pretrained_models/tsm_rgb_epic.ckpt
```
**Note:** Adjust the options based on your GPU resources and dataset size.

---
## Evaluation
To evaluate the model and generate predictions on the validation (or test) split, run:
```bash
python test_models.py Assembly101 \
  --weights "pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth.tar" \
  --test_segments 8 --batch_size 64 -j 16 --test_crops 1 \
  --text_file data/validation_combined.txt
```
### Output Files:
- `preds.npy`: Contains predicted fine-grained action labels for each segment (in the same order as in the text file).
- `scores.npy`: Contains the score vectors (1380-dimensional) for each segment.

---
## Additional Notes
- If using a different dataset split, update the `text_file` argument accordingly.
- Pretrained models must be downloaded separately and placed in `pretrained_models/`.
- Ensure that data paths in the scripts are correctly set before running the commands.

- 
## Issues & Troubleshooting

### Evaluation: Frames Alignment Issue
- Make sure the extracted frames are **correctly aligned** with the `.txt` annotation files.  
- The frame filenames should match the expected format in `image_tmpl` in the dataset class.
- If frames and `.txt` files are misaligned, update the dataset loading logic to match the correct frame indices.

### Missing or Incorrect Annotations
- If you encounter issues with missing annotations, ensure that the `gen_fine_labels.py` script ran successfully.
- Verify that the generated `train_combined.txt`, `validation_combined.txt`, and `test_combined.txt` files contain expected entries.

### Model Fails to Load Weights
- If the model does not load pretrained weights, check if the required checkpoint file is present in `pretrained_models/`.
- Ensure the `--tune_from` path is correct when fine-tuning.

### Performance Issues
- If training or evaluation is slow, reduce `--batch-size` or `--num_segments`.
- Ensure `num_workers (-j)` is properly set for optimal data loading performance.

---
## Citation
If you use this code or dataset in your research, please cite the Assembly101 dataset paper.

---
## License
This repository follows the original TSM repository's license.

For any issues, feel free to raise them in the repository or contact the maintainers.
