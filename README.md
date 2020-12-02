# Hateful Memes

A demo colab notebook is available [here](https://colab.research.google.com/drive/15CxLPwDXVS2ypPWP2eoV2RDbGwSpY61d?usp=sharing)

## Prerequisites

Download the zip file from [here](https://drive.google.com/file/d/1YJKmx4HaymEUBP-u93IFDT9UH-V1dCu5/view?usp=sharing)

Unzip files to hateful-memes/datasets

## Models and Methods

It supports 2 type of models, ViLBERT CC and Visual BERT COCO.

Methods include original, text augmentation (back translation), image captioning, and object labels.

| Model Key   | Method                                               | Config                                                    |
|-------------|------------------------------------------------------|-----------------------------------------------------------|
| vilbert     | original                                             | configs/vilbert_original.yaml                             |
| vilbert     | text augmentation                                    | configs/vilbert_back_translation.yaml                     |
| vilbert     | object labels                                        | configs/vilbert_objects.yaml                              |
| vilbert     | object labels + text augmentation                    | configs/vilbert_objects_back_translation.yaml             |
| vilbert     | image captioning                                     | configs/vilbert_caption.yaml                              |
| vilbert     | image captioning + text augmentation                 | configs/vilbert_caption_back_translation.yaml             |
| vilbert     | image captioning + object labels                     | configs/vilbert_caption_objects.yaml                      |
| vilbert     | image captioning + object labels + text augmentation | configs/vilbert_caption_objects_back_translation.yaml     |
| visual_bert | original                                             | configs/visual_bert_original.yaml                         |
| visual_bert | text augmentation                                    | configs/visual_bert_back_translation.yaml                 |
| visual_bert | object labels                                        | configs/visual_bert_objects.yaml                          |
| visual_bert | object labels + text augmentation                    | configs/visual_bert_objects_back_translation.yaml         |
| visual_bert | image captioning                                     | configs/visual_bert_caption.yaml                          |
| visual_bert | image captioning + text augmentation                 | configs/visual_bert_caption_back_translation.yaml         |
| visual_bert | image captioning + object labels                     | configs/visual_bert_caption_objects.yaml                  |
| visual_bert | image captioning + object labels + text augmentation | configs/visual_bert_caption_objects_back_translation.yaml |

## Start Training

`$ python tools/run.py config=<CONFIG> model=<MODEL KEY> dataset=hateful_memes`

This will save the training outputs to an experiment folder under `./save` directory.
