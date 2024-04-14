# Image-Captioning
This Project was adapted from [here](https://github.com/Dantekk/Image-Captioning/tree/main), and was used to demonstrate how open source projects can be adapted to other use cases fairly quickly with little modification. The following project explores the feasibility of using a CNN and Transformer encoder/decoder framework to provide captions to Chest X-ray Images. 

## Dataset 
The model has been trained on Frontal Chest X-Ray Images from the (NIH Open I Database)[https://openi.nlm.nih.gov/]
- Number of training images: 3054
- Number of testing images: 764

### My settings
For my training session, I left the default variables the original author (Dantekk)[https://github.com/Dantekk/Image-Captioning/tree/main] used.
```python
# Desired image dimensions
IMAGE_SIZE = (299, 299)
# Max vocabulary size
MAX_VOCAB_SIZE = 2000000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512
# Number of self-attention heads
NUM_HEADS = 6
# Per-layer units in the feed-forward network
FF_DIM = 1024
# Shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 512
# Batch size
BATCH_SIZE = 64
# Numbers of training epochs
EPOCHS = 14

# Reduce Dataset
# If you want reduce number of train/valid images dataset, set 'REDUCE_DATASET=True'
# and set number of train/valid images that you want.
#### COCO dataset
# Max number train dataset images : 68363
# Max number valid dataset images : 33432
REDUCE_DATASET = False
# Number of train images -> it must be a value between [1, 68363]
NUM_TRAIN_IMG = 68363
# Number of valid images -> it must be a value between [1, 33432]
# N.B. -> IMPORTANT : the number of images of the test set is given by the difference between 33432 and NUM_VALID_IMG values.
# for instance, with NUM_VALID_IMG = 20000 -> valid set have 20000 images and test set have the last 13432 images.
NUM_VALID_IMG = 20000
# Data augumention on train set
TRAIN_SET_AUG = True
# Data augmention on valid set
VALID_SET_AUG = False
# If you want to calculate the performance on the test set.
TEST_SET = True

### My results