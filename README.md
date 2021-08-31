# KM-BART: Knowledge Enhanced Multimodal BART for Visual Commonsense Generation (ACL 2021) 

### **Yiran Xing***, **Zai Shi***, **Zhao Meng***, **Gerhard Lakemeyer**, **Yunpu Ma**, **Roger Wattenhofer**   

∗The first three authors contribute equally to this work

[[Paper]](https://aclanthology.org/2021.acl-long.44.pdf) [[Supplementary]](https://aclanthology.org/attachments/2021.acl-long.44.OptionalSupplementaryMaterial.pdf)

![image](https://user-images.githubusercontent.com/14837467/118099580-b9b53a80-b3d5-11eb-9e86-188a99fd71d5.png)

![image](https://user-images.githubusercontent.com/14837467/118100120-62fc3080-b3d6-11eb-9196-a221024fc970.png)

## How to Cite Our Work

```
@inproceedings{KM-BART,
    title = "{KM}-{BART}: Knowledge Enhanced Multimodal {BART} for Visual Commonsense Generation",
    author = "Xing, Yiran  and
      Shi, Zai  and
      Meng, Zhao  and
      Lakemeyer, Gerhard  and
      Ma, Yunpu  and
      Wattenhofer, Roger",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "525--535"
}
```

## Installation

1. Clone the repository recursively
    ```
    git clone --recursive https://github.com/FomalhautB/KM-BART-ACL.git
    ```

2. Create conda environment
    ```
    conda env create -f environment.yaml
    ```

The following steps are only required for feature extraction.

1. Install `bottom-up-attention.pytorch`. Please refer to [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch/tree/1713806b18f4c3aaf5e5528377c4ed3db557bee7), for more details.
    ```bash
    cd bottom-up-attention.pytorch
    # install detectron2
    cd detectron2
    pip install -e .
    cd ..
    # install the rest modules
    python setup.py build develop
    cd ..
    ```

2. Install `comet-commonsense`. Please refer to [comet-commonsense](https://github.com/atcbosselut/comet-commonsense/tree/0042875b79af18a5b30c502613bd4a832cb47627) for more details.
    ```bash
    cd comet-commonsense
    # download data
    bash scripts/setup/get_atomic_data.sh
    bash scripts/setup/get_model_files.sh
    # install dependencies
    pip install tensorflow
    pip install ftfy==5.1
    conda install -c conda-forge spacy
    python -m spacy download en
    pip install tensorboardX
    pip install tqdm
    pip install pandas
    pip install ipython
    ```

## Data Preparation

### VCG
1. Download the images from [here](https://visualcommonsense.com/download/) and decompress the images into `$VCR_DATASET`   
2. Download the annotations from [here](https://visualcomet.xyz/dataset) and decompress the annotations into `$VCG_ANNOTATION`  
3. Extract features and save the features in `$VCG_DATA`:  
    ```bash
    python -m scripts.prepare_vcg \
        --data_dir $VCR_DATASET \ 
        --output_dir $VCG_DATA \
        --annot_dir $VCG_ANNOTATION \
        --gpu_num 4
    ```

### COCO  
1. Download the train images from [here](http://images.cocodataset.org/zips/train2014.zip) and decompress the images into `$COCO_TRAIN`
2. Download the validation images from [here](http://images.cocodataset.org/zips/val2014.zip) and decompress the images into `$COCO_VAL`
3. Download the annotations from [here](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) and decompress the annotations into `$COCO_ANNOTATION`
4. Extract features and save the features in `$COCO_DATA`:  
    ```bash
    python -m scripts.prepare_coco \
        --train_dir $COCO_TRAIN \
        --val_dir $COCO_VAL \
        --annot_dir $COCO_ANNOTATION  \
        --output_dir $COCO_DATA \
        --gpu_num 4
    ```

### SBU and CC
1. Download the json files for image urls and captions from [here](http://www.cs.virginia.edu/~vicente/sbucaptions/) and Decompress the two files into `$SBU_ANNOTATION`
2. extract the features, bounding box and labels, build image annotations and save into `$OUTPUT_DATA` (This will download the images first and save in `$SBU_DATA`):
    ```bash
    python -m scripts.prepare_sbu \
        --download \
        --data_dir $SBU_DATA \
        --output_dir $OUTPUT_DATA \
        --annot_dir $SBU_ANNOTATION \
        --gpu_num 4 \
        --n_jobs 8
    ```

### VG
1. Download the objects, relationships, region descriptions, attributs and image meta data from [here](https://visualgenome.org/api/v0/api_home.html) and decompress them into `$VG_ANNOTATION`
2. Download the images from the same link above and decompress them into `$VG_IMAGES`
    ```bash
    python -m scripts.prepare_vg \
        --annot_dir $VG_ANNOTATION \
        --output_dir $VG_DATA \
        --data_dir $VG_IMAGES \
        --gpu_num 4 \
    ```

### Reasoning (SBU and COCO)
1. Download the pretrained weight `atomic_pretrained_model.pickle` of COMET from [comet-commonsense](https://github.com/atcbosselut/comet-commonsense/tree/0042875b79af18a5b30c502613bd4a832cb47627)
    - Save it to `$LOAD_PATH`.
    - Follow the instructions in [comet-commonsense](https://github.com/atcbosselut/comet-commonsense/tree/0042875b79af18a5b30c502613bd4a832cb47627) to make the dataloader of COMET.
2. Download the json files for image urls and captions from [here](http://www.cs.virginia.edu/~vicente/sbucaptions/) and decompress the two files into `$SBU_ANNOTATION`.
3. Download the SBU dataset and save the images in `$SBU_DATA` and decompress the features, bounding box and labels of images and save into `$SBU_DATA`.
4. Generate inferences and save the inferences in `$REASON_DATA`.
   ```bash
   python -m scripts.prepare_sbu_reason \
        --output_dir $REASON_DATA \
        --annot_dir  $SBU_ANNOTATION \
        --model_file $LOAD_PATH/COMET \
        --gpu_num 2 \
        --sampling_algorithm topk-3
   
   # rename the output file
   mv $REASON_DATA/train.json $SBU_DATA/reason_train.json
   ```
4. Filter the newly generated inferences with a KM-BART pretrained on VCG (also in `$LOAD_PATH`) and save the final results in `$OUTPUT_DATA`.
   ```bash
   python -m scripts.filter_reason  \
        --data_dir $SBU_DATA \
        --output_dir $OUTPUT_DATA \
        --checkpoint $LOAD_PATH/KM-BART
   ```
   
  
## Training

### Pretrain from scratch
- Example of pretraining on COCO + SBU with 1 GPU and 4 CPUs from scratch (no pretrained weights)
    ```bash
    python pretrain \
        --dataset coco_train $COCO_DATA \
        --dataset coco_val $COCO_DATA \
        --dataset sbu_train $SBU_DATA \
        --checkpoint_dir $CHECKPOINT_DIR \
        --gpu_num 1 \
        --batch_size 32 \
        --master_port 12345 \
        --log_dir $LOG_DIR \
        --amp \
        --num_workers 4 \
        --model_config config/pretrain_base.json
    ```

### Pretrain from facebook bart-base
- Example of loading pretrained weights from facebook bart base and train on COCO
    ```bash
    python pretrain \
        --dataset coco_train $COCO_DATA \
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_config config/pretrain_base.json \
        --checkpoint facebook/bart-base
    ```

### Continue pretraining 
- Example of loading pretrained weights from previous checkpoint and continue to train on COCO
    ```bash
    python pretrain \
        --dataset coco_train $COCO_DATA \
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_config config/pretrain_base.json \
        --checkpoint $CHECKPOINT \
        --continue_training
    ```

### Train VCG
- Example of loading weights from pretrained checkpoint and fine tune on VCG. Validation will of loss and score will be done at the end of each epoch
    ```bash
    python vcg_train \
        --data_dir $VCG_DATA \
        --checkpoint_dir $CHECKPOINT_DIR \
        --validate_loss \
        --validate_score \
        --model_config config/vcg_base.json \
        --checkpoint $CHECKPOINT \
    ```

### Generate and evaluate VCG
- Example of generating sentences for VCG:  
    ```bash
    python vcg_generate \
        --data_dir $VCG_DATA \
        --checkpoint $CHECKPOINT \
        --output_file $GENERATED_FILE \
    ```

- Example of evaluating the generated file for VCG validation set:    
    ```bash
    python vcg_eval \
        --generation $GENERATED_FILE \
        --reference $VCG_DATA/val_ref.json
    ```

## Pretrained Weights
- [Model used in filtering](https://drive.google.com/file/d/1hZhMrhWSz74GSwp5Lq3R7qC4BqYBuzZL/view?usp=sharing)
- [Full model with event](https://drive.google.com/file/d/1YEnV7PJHHTj1CP9gZrafVVFkisggBQd7/view?usp=sharing)
- [Full model without event](https://drive.google.com/file/d/1YtUHKR85Qk9xCWpnI0j65A0yN8C9tIZ-/view?usp=sharing)
