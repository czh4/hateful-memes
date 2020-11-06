# Hateful Memes

A demo colab notebook is available [here](https://colab.research.google.com/drive/15CxLPwDXVS2ypPWP2eoV2RDbGwSpY61d?usp=sharing)

## Prerequisites

Download the zip file from [here](https://drive.google.com/file/d/1CvAApbvMS2TQVmrbwoalBXaRQfaHk0ik/view?usp=sharing)

Convert files `$ python tools/convert.py --zip_file='zip file location' --bypass_checksum=1 --mmf_data_folder='datasets'`

## Models and Methods

It supports 2 type of models, ViLBERT CC and Visual BERT COCO.

Methods include original, back translation, image captioning, and image augmentation.

| Model Method                   | Model Key   | Config                                    |
|--------------------------------|-------------|-------------------------------------------|
| vilbert + original             | vilbert     | configs/vilbert_original.yaml             |
| vilbert + back translation     | vilbert     | configs/vilbert_back_translation.yaml     |
| vilbert + image captioning     | vilbert     | configs/vilbert_caption.yaml              |
| visual bert + original         | visual_bert | configs/visual_bert_original.yaml         |
| visual bert + back translation | visual_bert | configs/visual_bert_back_translation.yaml |
| visual bert + image captioning | visual_bert | configs/visual_bert_caption.yaml          |

## Start Training

`$ python tools/run.py config=<CONFIG> model=<MODEL KEY> dataset=hateful_memes`

This will save the training outputs to an experiment folder under `./save` directory.
