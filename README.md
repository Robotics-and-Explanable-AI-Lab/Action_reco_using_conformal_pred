Assembly101 TSM Baseline – Replication Instructions
This repository contains the code to train, validate, and generate test set predictions for fine‐grained action recognition on the Assembly101 dataset using a Temporal Shift Module (TSM) network.

The implementation is adapted from the original TSM repository and has been modified to work with the Assembly101 annotations and data splits.

Overview
The Assembly101 dataset includes videos recorded from 12 synchronized camera views (8 static RGB and 4 egocentric monochrome) with fine‐grained action labels. This repository provides code for:

Data preparation: Generating TSM-style annotations from the provided CSV files.
Training: Fine-tuning a TSM model (e.g. starting from an EPIC-KITCHENS pretrained model) on Assembly101.
Evaluation: Generating predictions on validation or test splits and saving the outputs in NumPy files (preds.npy and scores.npy).
Requirements
Python 3.6+
PyTorch (>=1.6)
torchvision
NumPy
Other standard packages (PIL, tqdm, etc.)
It is recommended to use a virtual environment (e.g., conda or venv) and install the required packages.

Repository Structure
main.py: Training and validation script.
test_models.py: Evaluation script that loads a pretrained model and generates predictions.
gen_fine_labels.py: Script to convert the provided CSV annotations into TSM-style text files.
ops/: Contains dataset classes and model definitions.
pretrained_models/: Pretrained TSM model checkpoints (download as needed).
data/: Contains the TSM-style annotation files and the category.txt file listing the 1380 fine-grained actions.
Data Preparation
Annotation Generation:

Convert the CSV annotations into TSM-style text files using the provided script. For example, to generate annotations for the combined modality on the test split:

bash
Copy
python gen_fine_labels.py combined test
(Use rgb or mono if you wish to work with a single modality.)

Directory Structure for Frames:

Make sure your extracted frames follow the naming convention expected by the dataset class. For example, if you are using the RGB modality, each camera’s folder should contain frames named like:

Copy
C10095_rgb_frame_000086.jpg
Adjust the image_tmpl in the dataset code if your naming is different.

Data Splits:

Place your train_combined.txt, validation_combined.txt, and test_combined.txt (or a subset thereof for replication) in the data/ folder.

Training
To fine-tune the model on Assembly101, run the training script. For example, to fine-tune starting from an EPIC-KITCHENS pretrained model:

bash
Copy
python main.py Assembly101 combined --arch resnet50 --num_segments 8 --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb --tune_from pretrained_models/tsm_rgb_epic.ckpt
Note: The options may be adjusted depending on your GPU resources and dataset size.

Evaluation
To evaluate the model and generate predictions on the validation (or test) split, run:

bash
Copy
python test_models.py Assembly101 --weights "pretrained_models/TSM_Assembly101_combined_resnet50_shift8_blockres_avg_segment8_e50.pth.tar" --test_segments 8 --batch_size 64 -j 16 --test_crops 1 --text_file data/validation_combined.txt
This will output two files:

preds.npy: Contains the predicted fine-grained action labels for each segment (in the same order as in the text file).
scores.npy: Contains the score vectors (1380-dimensional) for each segment.
