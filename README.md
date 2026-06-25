Deep Learning Applications — Laboratories
=========================================

---

Welcome to the repository for the **Deep Learning Applications** course laboratories. This collection contains three hands-on projects that tackle various Deep Learning challenges: from image classification and object detection, to multimodal text-to-image retrieval, and finally autonomous agent control via Deep Reinforcement Learning.

Each directory contains its own `README.md` with detailed analysis, experiment results, and instructions for running the code. Below is a brief introductory overview of the three laboratories.

🚦 Lab 1: Image Classification and Object Detection (Faster R-CNN)
------------------------------------------------------------------

The first laboratory focuses on computer vision applied to traffic signs. We begin with image classification on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset, exploring the use of pre-trained CNNs as feature extractors and through end-to-end **Fine-Tuning** techniques. The project then advances to **Object Detection**: we implement a **Faster R-CNN** architecture where the original backbone is replaced by the model previously fine-tuned on the GTSRB dataset, allowing us to evaluate the effectiveness of transfer learning in a simultaneous localization and classification task.

![Detection Example](Lab%201/img/detection_example.png)

---

🔍 Lab 2: Text-to-Image Retrieval with CLIP
-------------------------------------------

The second laboratory explores the potential of multimodal Vision-Language models by building a fully functional visual search engine. Leveraging the power of the **CLIP** (Contrastive Language-Image Pretraining) model and the **Flickr8k** dataset, this project aligns text and image vector spaces. By computing the _cosine similarity_ between the extracted embeddings, the system is capable of retrieving the best-matching images for a given text query (Zero-Shot Retrieval). The laboratory culminates in the development of an interactive web user interface built with **Gradio**.

![Gradio Interface](Lab%202/img/gradio_interface.png)

---

🤖 Lab 3: Deep Reinforcement Learning (REINFORCE and DQN)
---------------------------------------------------------

The third and final laboratory is a practical study of Deep Reinforcement Learning algorithms, covering both _Policy Gradient_ and _Value-based_ methods. The project involves training agents in simulated environments (such as **CartPole-v1** and **LunarLander-v3**) by building a modular custom package (`drl_lab`) from scratch. The implementation ranges from an improved **REINFORCE** algorithm with standardization, to an **Actor-Critic** architecture with a learned value baseline, concluding with a full **DQN** agent complete with an experience replay buffer and target networks.

|       |        |
| :---: | :---: |
| ![](Lab%203/img/cart_pole.gif) | ![](Lab%203/img/lunar_lander.gif) |

---

## Author

**Tommaso Pastorelli** — Università degli Studi di Firenze  
`tommaso.pastorelli1@edu.unifi.it`
