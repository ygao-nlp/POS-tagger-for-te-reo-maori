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

### Prepare Data
All data is stored in three .txt files, where maori-tag-string(part).txt is TRM, maori-tag-string.txt is extended TRM and maori-tag-string-noisy.txt is the noisy data part of extended TRM
### Preprocess Data
The data need to be processed and the format of the processed data is as follows:

[(E, TAM), (pai, N), (ana, TAM), (a, DET), (tom, N), (?, PUNCT)]

In addition, the information needed for models is returned, which is about the dataset such as word dictionaries, tag dictionaries, processed data and so on. Finally, the string type data is replaced with numbers based on the built dictionary. 
For the UDPOS dataset, which is derived from torchtext [1], it provides many built-in functions, these functions can be used directly to meet requirements.
All the code and functions for preprocessing data are in maori_data.py. The helper and transformer functions are responsible for cleaning the raw data and processing it into the needed format. TEM, extend_TEM, noisy_TEM use the two mentioned functions to process the data. MaoriDataset processes the data into digital sequences in order to batch data.

## Building Models

The model and the related code are placed in model.py. There are four models, including random forest, CRF, BiLSTM, and BiLSTM-CRF. Random forest and CRF are from the sklearn library [2], and the formatted data can be used directly. BiLSTM-CRF is based on pytorch tutorial [3] and has been partially modified.

For the random forest model, we used only simple contextual features, which included the word, the previous word, the word before the previous word, the word two before the previous word and the following word. These features can be adapted based on the data, but it is important to ensure that the features are represented by numbers.

For the CRF model, we add morphology-based features to each word in the input sentence, including the word, whether it's the first word in the sentence, whether it's the last word in the sentence, whether it has a hyphen, whether it's a number, the previous word and the following word. All of these features are present as a number. As with random forest, more features can be added to achieve a better result.

For BiLSTM, the architecture of the model is an embedding layer plus a BiLSTM layer plus a fully connected layer. The embedding dimension is 100 and the hidden layer dimension is 32. We have also added dropout to the embedding layer, the bilstm layer, and the dropout rate of 0.3. These parameters are flexible and can be changed by modifying the variables in the training file. Alternatively, N_LAYERS can be changed to a higher number so that the model has more BiLSTM layers.

For BiLSTM-CRF, we have replaced the fully connected layer with a CRF layer. Otherwise, it is used in the same way as the BiLSTM

## Result
Models are trained and evaluated in four other files, each file corresponding to one of the following experiments. The current results are obtained from the model through 50 iterations, and we believe that a higher number of iterations will lead to better results.

Based on the results we assumed that data with noisy (i.e. incorrect words with the same usage) has a negative effect on the CRF. To test this assumption, we trained models with noisy part of extended TRM and then tested them with TRM. In this experiment, it was clear that the results of CRF-based taggers were much worse than the BiLSTM model, which means that the CRF model does not handle noisy data well. 
Overall, all results of our taggers outperform the random forest. For the CRF, it usually requires many more iterations to complete converge. This is probably the main factor that affects the performance of CRF and BiLSTM-CRF. However, more iterations mean longer training time and we were not able to keep the BiLSTM-CRF trained on the extended TRM and UDPOS datasets until the best results were achieved. Based on our results, BiLSTM-CRF is more suitable for small size datasets, and BiLSTM-based tagger provides faster speeds when working with large size datasets.


|Training set|Test set|Random Forest|CRF|BiLSTM|BiLSTM-CRF|
|------------|------------|----------|----------|----------|----------|
|TRM|TRM|83.98|93.07|98.23|98.93|
|Extended TRM|Extended TRM|97.44|98.58|98.74|98.60|
|Extended TRM (Noisy part)|TRM|88.65|90.96|94.07|91.78|
|UDPOS|UDPOS|75.73|86.56|91.21|90.09|

## Contact

If you have any problem or encounter mysterious things during simulating this code, contact me by sending email to 0211734456gy@gmail.com

## References

[1] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d\textquotesingle Alch\'{e}-Buc, E. Fox, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

[2]	Pedregosa, F., Varoquaux, Ga"el, Gramfort, A., Michel, V., Thirion, B., Grisel, O., … others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825–2830.

[3]	Seung-won Park. (2019). ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF. https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
