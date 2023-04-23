<!-- ABOUT THE PROJECT -->
<br />
<div align="center">
  <a href="">
    <img src="oticon_header.png" alt="Logo">
  </a>

  <h3 align="center">2D CNN Sound Classifier by Pew Pew Sounds</h3>

</div>

## Oticon Audio Explorers Challenge 2023

The CNN architecture used in this implementation consists of two convolutional layers, each followed by a max-pooling layer and dropout for regularisation. After the convolutional layers, the output is flattened and passed through a fully connected layer with 64 units and a dropout layer. Finally, a softmax activation function is applied to the output layer, which consists of as many units as there are classes in the dataset.

<!-- GETTING STARTED -->
## Getting Started

Install python, we used version 3.10.11, and the packages listed below:

* numpy
* tensorflow
* scikit-learn

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/andreaslborg/OticonChallenge2023
   ```
2. Install packages
   ```sh
   pip install numpy tensorflow scikit-learn
   ```
3. Run and train the model on all the data and predict on the test data
   ```sh
   python .\CNN2D_100.py
   ```
4. Run and train the model on 70% of the data and validate on 30%
   ```sh
   python .\CNN2D_70.py
   ```

<!-- Authors -->
## Authors

* <a href="https://www.linkedin.com/in/alexanderlborg/">Alexander Løvig Borg</a>
* <a href="https://www.linkedin.com/in/andreaslborg/">Andreas Løvig Borg</a>
* <a href="https://www.linkedin.com/in/anton-egholm/">Anton Sig Egholm</a>
