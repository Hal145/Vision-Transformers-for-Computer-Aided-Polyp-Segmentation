# Vision-Transformers-for-Computer-Aided-Polyp-Segmentation

Input Encoding: Convert the input image into a set of patches. Each patch will be represented as a flattened vector. Additionally, each patch will have a learnable positional embedding.

Patch Embedding: Apply a linear projection to the flattened patch vectors, followed by a positional embedding addition. This step maps the patches to a lower-dimensional representation, which will be the input to the transformer layers.

Transformer Encoder: Stack several Transformer Encoder layers. Each Transformer Encoder layer consists of a multi-head self-attention mechanism followed by a feed-forward neural network. This step allows the model to capture the spatial relationships and patch dependencies.

Decoder: After the Transformer Encoder layers, we can use a decoder to upsample the features and generate the final segmentation mask. A common choice for the decoder is a series of upsampling convolutional layers, which gradually increase the spatial resolution.
