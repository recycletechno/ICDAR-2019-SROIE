# ICDAR-2019-SROIE

## Competition tasks

A take in ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE)

Link to original competition - [https://arxiv.org/pdf/2103.10213.pdf](https://arxiv.org/pdf/2103.10213.pdf)

Scanned receipts OCR and key information extraction (SROIE) represent the processeses of recognizing text from scanned receipts and extracting key texts from them and save the extracted tests to structured documents

Competition consists of 3 tasks:
- Scanned Receipt Text Localisation
- Scanned Receipt OCR
- Key Information Extraction from Scanned Receipts

For **Task 1, 2** each image in the dataset should be annotated with text bounding boxes (bbox) and the transcript of each text bbox. Locations are annotated as rectangles with four vertices, which are in clockwise order starting from the top. Annotations for an image are stored in a text file with the same file name. File format:

```
x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1,transcript_1
x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2,transcript_2
x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3,transcript_3
…
x1_N,y1_N,x2_N,y2_N,x3_N,y3_N,x4_N,y4_N,transcript_N
```
The aim of **Task 3** is to extract texts of a number of key fields from given receipts, and save the texts for each receipt image in a json file. Each image in the dataset should be also annotated and stored with a text file with following format:

```json
{
  "company": "STARBUCKS STORE #10208",
  "address": "11302 EUCLID AVENUE, CLEVELAND, OH (216) 229-0749",
  "date": "14/03/2015",
  "total": "4.95"
}
```

## Methods and Models

This attempt was undertaken during 5 days Hackaton in Jan 2023 on master's degree program from Ural Federal University. Main goal was to build a baseline code in a short time and get satisfactory results. First I made CPU version of code, then made this work on GPU therefore speed increased significantly. Further plans: apply methods for improving final metrics with transfer learning, nltk etc.

### Task 1, 2: PaddleOCR

I used out of the box solution with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and got good results to start. Best results were on **en_PP-OCRv3_rec** as a recognition model and **en_PP-OCRv3_det** as a detection model. Did not use angle classifier as no text is rotated by 180 degrees, used cls=False to get better performance

### Task 3: DonutProcessor

The Donut model was proposed in [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664). Donut consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform document understanding tasks such as document image classification, form understanding and visual question answering. Overview, code examples as a huggingface transformer available [here](https://huggingface.co/docs/transformers/main/en/model_doc/donut)

