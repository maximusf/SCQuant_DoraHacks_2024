# SC Quantathon 2024 - DoraHacks Quantum Random Number Generator Challenge

## Project Overview

Welcome to our submission for the **DoraHacks Quantum Random Number Generator (QRNG) Challenge** as part of the **SC Quantathon 2024**. This project focuses on implementing quantum random number generation using Quantum Processing Units (QPUs) and exploring various techniques to analyze, classify, and enhance the generated random numbers. Our team is excited to dive into the cutting-edge field of quantum computing and contribute to advancements in true random number generation.

## Challenge Summary

The DoraHacks challenge centers on **Quantum Random Number Generation (QRNG)**, a technique that utilizes the inherent unpredictability of quantum mechanics to generate true randomness. As hacking algorithms and artificial intelligence (AI) continue to evolve, the need for high-entropy random number generation becomes even more critical for securing cryptography systems.

The challenge is divided into five stages, ranging from implementing QRNG on quantum devices to classifying random number data using machine learning, analyzing quantum noise, and ultimately applying QRNG in real-world encryption protocols.

## Stages of the Challenge

### Stage 1: Implementing a QRNG
- **Objective**: Generate quantum random numbers using a QPU.
- **Description**: Create and run quantum circuits to produce QRNG data. Analyze and visualize the results.
- **Resources**: IBM Qiskit tutorials, QRNG circuit examples.

### Stage 2: Achieving High Accuracy with Classification Models
- **Objective**: Train machine learning models to classify QRNG and classical random numbers.
- **Description**: Achieve over 98% accuracy in classifying quantum vs classical random numbers.
- **Resources**: Gradient boosting model skeleton code, sample datasets.

### Stage 3: Characterizing Noise and Fidelity
- **Objective**: Analyze the noise and fidelity of the quantum machine used for QRNG.
- **Description**: Compare noise patterns of quantum simulators and real QPUs and create data visualizations.
- **Resources**: IBM QPU error rate data.

### Stage 4: Pre-processing and Post-processing for High Entropy
- **Objective**: Enhance the entropy of QRNG data using pre- and post-processing techniques.
- **Description**: Test the randomness of QRNG data using the NIST Statistical Test Suite.
- **Resources**: Randomness extraction functions, SciPy tools, NIST test suite.

### Stage 5: Measuring Entropy and Real-world Implementation
- **Objective**: Measure and compare the entropy of QRNG data with PRNG and TRNG data.
- **Description**: Implement QRNG in a real-world application such as encryption.
- **Resources**: Entropy measurement tools, application integration examples.

## Getting Started

### Prerequisites
To run the QRNG circuits and classification models, you'll need the following software:
- [Qiskit](https://qiskit.org/) for building and running quantum circuits
- Python 3.8 or later
- Machine learning libraries such as **scikit-learn** and **XGBoost**

## Team Members
- Kevin Do
- Tyler Shaughnessy
- Maximus Fernandez

## Submission Guidelines
Our submission includes:
- A detailed description of our approach and results for each stage
- The source code available in this GitHub repository
- A demo video showcasing our work
- Visualizations of our analysis

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
We would like to thank **DoraHacks** and **SC Quantathon 2024** for organizing this challenge and providing valuable resources. Special thanks to the open-source quantum computing community for their tutorials and research.
