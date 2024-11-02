# Assignment 2 - Tova Erbén

## Data

15,274 images, 416x416 pixles

## Bonus A: make the in-class example actually learn something

I tried to increase the accuracy of the base code by doing the following:
* decreasing the kernel window in the maxpool2d layer (and adjusting the other values)
* decreasing the learning rate from 0.01 to 0.001 after learning that is the default lr for the Adam optimizer
* incresing the dropout rate from 0.01 to 0.3 (having seen recommended dropout ranges between 0.2-0.5)
* using the validation set that comes with the wikiart dataset while training the model,
  
I thought some of these changes made a big change in the accuracy (up to 8 %) but realized a bit too late that there is too much randomness in how the model is trained and evaluated, leading to different accuracy scores every time I ran the base code's `test.py` class.

## Part 1: fix class imbalance

`du --inodes`

`train`

18      ./Action_painting
15      ./Analytical_Cubism
688     ./Art_Nouveau_Modern
721     ./Baroque
268     ./Color_Field_Painting
91      ./Contemporary_Realism
390     ./Cubism
246     ./Early_Renaissance
1127    ./Expressionism
163     ./Fauvism
198     ./High_Renaissance
2269    ./Impressionism
246     ./Mannerism_Late_Renaissance
212     ./Minimalism
373     ./Naive_Art_Primitivism
41      ./New_Realism
413     ./Northern_Renaissance
80      ./Pointillism
244     ./Pop_Art
946     ./Post_Impressionism
1712    ./Realism
369     ./Rococo
1157    ./Romanticism
679     ./Symbolism
38      ./Synthetic_Cubism
197     ./Ukiyo_e


`test`
17      ./Abstract_Expressionism
2       ./Analytical_Cubism
32      ./Art_Nouveau_Modern
34      ./Baroque
6       ./Color_Field_Painting
3       ./Contemporary_Realism
15      ./Cubism
9      ./Early_Renaissance
52      ./Expressionism
8       ./Fauvism
9      ./High_Renaissance
101     ./Impressionism
12      ./Mannerism_Late_Renaissance
10      ./Minimalism
19      ./Naive_Art_Primitivism
1       ./New_Realism
23      ./Northern_Renaissance
6       ./Pointillism
15      ./Pop_Art
51      ./Post_Impressionism
82      ./Realism
13      ./Rococo
51      ./Romanticism
44      ./Symbolism
1       ./Synthetic_Cubism
12      ./Ukiyo_e


## Part 2: autoencode and cluster represenations

### Implementation 

For the autoencoder part I implemented an encoder  with:
* conv2d and ReLu activation layers (if time permitted I would also have tried adding some maxpooling layers)
* a decoder that does the reverse with `ConvTranspose2d` layers
* changed the existing train function to use `MSEloss()` (mean squared error loss) to measure the reconstruction error 
* changed the existing train function to take the input image (X) as the truth when meassuring the loss
* unlike the classification task in part 1, the loss dropped considrably during training (possibly due to less complex task)
* skipped using the validation data for this part, but that could possibly have improved it even further.

### Measure

For measuring the model's ability to reconstruct images I just:
* averaged the mean sqaured error over test set
But that number didn't feel very informative (0.0044) compared to the accuracy measuremnet of the previous task. The visual inspection of a single input/output image in the test set provided a better picture of the model's abilty:

![image](images/a2_input_output2.png)


### Cluster visualization

For the clustering and its visualization, I:
* transformed the encoded images from 4D to 2D (not sure if I did that the right way) since the `Kmeans` clustering method demanded that 
* applied Principal Component Analysis for further dimensionality reduction
* used matplotlib to cluster based on actual labels as well as PCA-based cluster labels.

![image](images/pca_encoded_images_actual_and_cluster_labels.png)

Judging by this visaluzation, the PCA clustering does not seem to perform terribly well.

## Part 3: generation/style transfer
