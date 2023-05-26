# Quantum Convolutional Neural Networks for Multi-Channel Supervised Learning
![architecture_redo_final](https://github.com/anthonysmaldone/QCNN-Multi-Channel-Supervised-Learning/assets/124306057/7b5f4497-8dfa-4271-82e8-fcae7341a5de)
## Summary
As the rapidly evolving field of machine learning continues to produce incredibly useful tools and models, the potential for quantum computing to provide speed up for machine learning algorithms is becoming increasingly desirable. In particular, quantum circuits in place of classical convolutional filters for image detection-based tasks are being investigated for the ability to exploit quantum advantage. However, these attempts, referred to as quantum convolutional neural networks (QCNNs), lack the ability to efficiently process data with multiple channels and therefore are limited to relatively simple inputs. In this work, we present a variety of hardware-adaptable quantum circuit ansatzes for use as convolutional kernels, and demonstrate that the quantum neural networks we report outperform existing QCNNs on classification tasks involving multi-channel data. We envision that the ability of these implementations to effectively learn interchannel information will allow quantum machine learning methods to operate with more complex data.

## Pre-print
link
## Usage
1) If running for the first time, execute ```python create_noisy_colors.py```  to create synthetic data
2) Run ```python train.py``` to see menu options and train QCNN
3) View results in the ```./output``` folder. Files are named based on the time the training for the model began
