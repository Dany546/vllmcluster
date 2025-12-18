Align DINO to CLIP
1.Training methods:
        a. Compare with mean text embeddings
        b. Contrastive learning with training augmentations and/or captions
        d. Use similarities between CLIP's text embeddings to define the pseudo-labels (either single embedding or average of embeddings)
        e. Use graph-based clustering methods instead of pseudo-labels-based CE loss
        f. Use inter-caption similarities to condition the temperature of the CE loss, same for images and their augmented versions
              Use inter-cluster and intra-clusters as in https://arxiv.org/abs/2307.11227?utm_source=chatgpt.com
                    Use text to define the pseudo-labels + images as supplementary data-wise information
2. Deep clustering: methods
        a. k-means, spectral k-means, agglomerative, GMM
        b. DBSCAN
        c. VQ-VAE
3. Deep clustering: ablation
        a. DINO space (ablation: is clip helpfull ?)
        b. DINO2CLIP space
        c. CLIP space (ablation: what is brought from images?)
        d. MLP bottleneck (lower-dim intermediate space)
4. vary MLP bottleneck
