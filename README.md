# Assignment 2 - Tova Erbén

## Pre-req

- `conda deactivate`

## Data

15,274 images, 416x416 pixles

## Bonus A: make the in-class example actually learn something

I tried to increase the accuracy of the base code by doing the following (see commit tag `204e011a80b4fce32107ee36d52a6d1fedd10351`):
* decreasing the kernel window in the maxpool2d layer (and adjusting the other values)
* incresing the dropout rate from 0.01 to 0.3 (having seen recommended dropout ranges between 0.2-0.5).
The kernel window change made no difference, but the increase in dropout improved the overall accuracy to 7-8 %.

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


