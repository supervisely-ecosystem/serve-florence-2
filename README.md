<div align="center" markdown>
<img src="https://github.com/user-attachments/assets/54112ea8-8ae3-42fa-906c-2ba53f552884"/>  

# Serve Florence-2
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#Model-application-examples">Model application examples</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-florence-2)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-florence-2)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-florence-2.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-florence-2.png)](https://supervise.ly)
 
</div>

# Overview

Florence is a foundation model designed for multimodal vision tasks, enabling unified handling of image analysis and text interaction. It employs a seq2seq transformer architecture to handle diverse tasks such as object detection, segmentation, image captioning, and visual grounding. The model introduces a unified approach to vision-language tasks, where textual prompts guide the model to produce task-specific output.

Florence processes visual data using a vision encoder that converts images into token embeddings. These embeddings are combined with textual prompts and passed through a multimodal encoder-decoder to generate outputs.

![operating principle 1](https://github.com/user-attachments/assets/ec79e92c-4699-41a2-babb-177877e768f2)

The model is trained on FLD-900M, a large dataset of over 900 million image-text pairs with detailed annotations for global, regional, and pixel-level tasks. 

Florence serves as a versatile tool capable of performing tasks such as image captioning, object detection, and segmentation through a single, unified architecture.

![florence-tasks](https://github.com/user-attachments/assets/95164496-e865-4ddc-8ed4-717e6bbeac39)

# How To Run

**Step 1** Select pretrained model architecture and press the **Serve** button.

![serve](https://github.com/user-attachments/assets/b59dfbb3-3c2c-47dd-b920-147cf584c214)

**Step 2.** Wait for the model to deploy.

![deployed](https://github.com/user-attachments/assets/f9c20b41-40ea-4aeb-8e0d-e510c2f24bc5)

# Model application examples
