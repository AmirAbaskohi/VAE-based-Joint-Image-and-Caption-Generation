# VAE-based Joint Image and Caption Generation

This repository contains my research internship code at the University of Illinois Chicago under the supervision of Prof. Pedram Rooshenas.

Our goal was to create a joint generative model based on Variational Auto Encoder Models. Not only the output is interesting but also the join distribution in the 
latent space may result in better-generated image quality.

I Learned the concept of VAE using <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=video&cd=&cad=rja&uact=8&ved=2ahUKEwid6IKdiIr5AhVw57sIHSHJCNIQtwJ6BAgLEAI&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DuaaqyVS9-rM&usg=AOvVaw2UmuS4u7uWek7o7Kg7Gb8f">
this video</a>.

## Variational Auto Encoder

In machine learning, a variational autoencoder, is an artificial neural network architecture introduced by Diederik P. Kingma and Max Welling,
belonging to the families of probabilistic graphical models and variational Bayesian methods.

![image](https://user-images.githubusercontent.com/50926437/180223287-e7edbcc4-8fe2-4ce3-8879-cc95d9432ce1.png)


## NVAE
NVAE, or Nouveau VAE, is deep, hierarchical variational autoencoder. It can be trained with the original VAE objective, unlike alternatives such as VQ-VAE-2. NVAE’s design focuses on tackling two main challenges: (i) designing expressive neural networks specifically for VAEs, and (ii) scaling up the training to a large number of hierarchical groups and image sizes while maintaining training stability.

To tackle long-range correlations in the data, the model employs hierarchical multi-scale modelling. The generative model starts from a small spatially arranged latent variables as z1

and samples from the hierarchy group-by-group while gradually doubling the spatial dimensions. This multi-scale approach enables NVAE to capture global long-range correlations at the top of the hierarchy and local fine-grained dependencies at the lower groups.

Additional design choices include the use of residual cells for the generative models and the encoder, which employ a number of tricks and modules to achieve good performance, and the use of residual normal distributions to smooth optimization. See the components section for more details.

![image](https://user-images.githubusercontent.com/50926437/180224019-e2028c91-a590-4c41-99c0-e42429299c01.png)


You can their paper which is available in the repo and in <a href="https://arxiv.org/pdf/2007.03898v3.pdf">this link</a>.



## Squeeze-and-Excitation Block
![image](https://user-images.githubusercontent.com/50926437/180224398-e00bc6da-c5e8-4f1c-85e1-746b6a50d758.png)

The Squeeze-and-Excitation Block is an architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise feature recalibration. The process is:

* The block has a convolutional block as an input.
* Each channel is "squeezed" into a single numeric value using average pooling.
* A dense layer followed by a ReLU adds non-linearity and output channel complexity is reduced by a ratio.
* Another dense layer followed by a sigmoid gives each channel a smooth gating function.
* Finally, we weight each feature map of the convolutional block based on the side network; the "excitation".

Their paper is available <a href="https://arxiv.org/abs/1709.01507v4">here</a>.

## Join Text and Image Embedding
I used `SEJE` which is a prototype for the paper Learning Text-Image Joint Embedding for Efficient Cross-Modal Retrieval with Deep Feature Engineering.

Overview: SEJE is a two-phase deep feature engineering framework for efficient learning of semantics enhanced joint embedding, which clearly separates the deep feature engineering in data preprocessing from training the text-image joint embedding model. We use the Recipe1M dataset for the technical description and empirical validation. In preprocessing, we perform deep feature engineering by combining deep feature engineering with semantic context features derived from raw text-image input data. We leverage LSTM to identify key terms, deep NLP models from the BERT family, TextRank, or TF-IDF to produce ranking scores for key terms before generating the vector representation for each key term by using word2vec. We leverage wideResNet50 and word2vec to extract and encode the image category semantics of food images to help semantic alignment of the learned recipe and image embeddings in the joint latent space. In joint embedding learning, we perform deep feature engineering by optimizing the batch-hard triplet loss function with soft-margin and double negative sampling, taking into account also the category-based alignment loss and discriminator-based alignment loss. Extensive experiments demonstrate that our SEJE approach with deep feature engineering significantly outperforms the state-of-the-art approaches.

![image](https://user-images.githubusercontent.com/50926437/180224937-9490dd59-8d93-4e4f-a49a-91d393caf431.png)


## How to run?

First you need to install requirements using:
```
pip3 install -r requirements.txt
```

### Running the embedding

#### Word2Vec
Training word2vec with recipe data:

- Download and compile [word2vec](https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip)
- Train with:

```
./word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 10 -threads 20 -train tokenized_text.txt -output vocab.bin
```
The pre-trained word2vec model can be found in [vocab.bin](https://drive.google.com/file/d/1Qu2tiLPlCu9KaR2vhAc4T2dZlvPrKXAn/view?usp=sharing).



#### Training
- Train the model with: 
```
CUDA_VISIBLE_DEVICES=0 python train.py 
```
We did the experiments with batch size 100, which takes about 11 GB memory.



#### Testing
- Test the trained model with
```
CUDA_VISIBLE_DEVICES=0 python test.py
```
- The results will be saved in ```results```, which include the MedR result and recall scores for the recipe-to-image retrieval and image-to-recipe retrieval.
- Our best model trained with Recipe1M (TSC paper) can be downloaded [here](https://drive.google.com/drive/folders/1q4MpqSXr_ZCy2QiBn1XV-B6fFlQFjwSV?usp=sharing).

### Running NVAE
```
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python3 train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset {datasetname} --batch_size 200 \
        --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 2 --use_se --res_dist --fast_adamax 
```
