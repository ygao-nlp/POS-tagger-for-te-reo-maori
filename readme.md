# Part-of-Speech Tagging for Te Reo Māori

This project uses three methods, CRF, BiLSTM and BiLSTM-CRF, to achieve lexical annotation of Maori. Three datasets were used, TRM, extended TRM and UDPOS from PyTorch, where TRM and extended TRM were the manually labelled datasets. Ultimately, the results of the experiments were evaluated by accuracy.

## Environment

### Hardware
* GeForce GTX 1080
### Software
#### OS
I used both Windows and Linux
#### Python Package
You have to install these packages before executing this code.
* Python >= 3.6
*	pytorch == 1.6.0
*	numpy == 1.18.1
*	sklearn == 0.22.1
*	sklearn_crfsuite == 0.3

## Usage

Prepare Data
All data is stored in three .txt files, where maori-tag-string(part).txt is TRM, maori-tag-string.txt is extended TRM and maori-tag-string-noisy.txt is the noisy data part of extended TRM
Preprocess Data
The data need to be processed and the format of the processed data is as follows:
[(E, TAM), (pai, N), (ana, TAM), (a, DET), (tom, N), (?, PUNCT)]
In addition, the information needed for models is returned, which is about the dataset such as word dictionaries, tag dictionaries, processed data and so on. Finally, the string type data is replaced with numbers based on the built dictionary. 
For the UDPOS dataset, which is derived from torchtext [1], it provides many built-in functions, these functions can be used directly to meet requirements.
All the code and functions for preprocessing data are in maori_data.py

## Building Models

The model and the code associated with it are placed in model.py. There are four models, including random forest, CRF, BiLSTM, and BiLSTM-CRF. Random forest and CRF are from the sklearn library [2], and the formatted data can be used directly. BiLSTM-CRF is based on pytorch tutorial [3] and has been partially modified.
Result
Models are trained and evaluated in four other files, each file corresponding to one of the following experiments. The current results are obtained from the model through 50 iterations, and we believe that a higher number of iterations will lead to better results.

Training set	Test set	Random Forest	CRF	BiLSTM	BiLSTM-CRF
TRM	        TRM	        83.98	        93.07	98.23	98.93
Extended TRM	Extended TRM	97.44	        98.58	98.74	98.60
Extended TRM    TRM	        88.65	        90.96	94.07	91.78
(Noisy part)	
UDPOS	        UDPOS	        75.73	        86.56	91.21	90.09

## Contact
---

If you have any problem or encounter mysterious things during simulating this code, contact me by sending email to 0211734456gy@gmail.com

## References
---

[1]	https://pytorch.org/text/stable/index.html
[2]	https://scikit-learn.org/stable
[3]	https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
