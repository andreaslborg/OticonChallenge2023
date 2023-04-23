<!-- PROJECT SHIELDS -->
<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="">
    <img src="oticon_header.png" alt="Logo">
  </a>

  <h3 align="center">CNN 2D Sound Classifier by Pew Pew Sounds</h3>

</div>



<!-- ABOUT THE PROJECT -->
## Oticon Audio Explorers Challenge 2023

The CNN architecture used in this implementation consists of two convolutional
layers, each followed by a max-pooling layer and dropout for regularisation. After
the convolutional layers, the output is flattened and passed through a fully connected
layer with 256 units and a dropout layer. Finally, a softmax activation function is
applied to the output layer, which consists of as many units as there are classes in
the dataset.

<!-- GETTING STARTED -->
## Getting Started

Install python, we used version 3.10.11, and the packages listed in prerequisites.

### Prerequisites

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
3. Run and train the model
   ```sh
   python .\CNN2D_functions_100.py
   ```


<!-- Authors -->
## Authors

* <a href="https://github.com/aborg123">Alexander Løvig Borg</a>
* <a href="https://github.com/andreaslborg">Andreas Løvig Borg</a>
* <a href="https://github.com/Anantonon">Anton Sig Egholm</a>
