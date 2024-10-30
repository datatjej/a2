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



