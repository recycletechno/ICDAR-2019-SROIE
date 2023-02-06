# ICDAR-2019-SROIE

A take in ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE)

Link to original competition - [https://arxiv.org/pdf/2103.10213.pdf](https://arxiv.org/pdf/2103.10213.pdf)

Scanned receipts OCR and key information extraction (SROIE) represent the processeses of recognizing text from scanned receipts and extracting key texts from them and save the extracted tests to structured documents

Competition consists of 3 tasks:
- Scanned Receipt Text Localisation
- Scanned Receipt OCR
- Key Information Extraction from Scanned Receipts

For Task 1, 2 each image in the dataset should be annotated with text bounding boxes (bbox) and the transcript of each text bbox. Locations are annotated as rectangles with four vertices, which are in clockwise order starting from the top. Annotations for an image are stored in a text file with the same file name. File format:

```
x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1,transcript_1
x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2,transcript_2
x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3,transcript_3
…
x1_N,y1_N,x2_N,y2_N,x3_N,y3_N,x4_N,y4_N,transcript_N
```

For the information extraction task, each image in the dataset is also annotated and stored with a text file with following format:

```json
{
  "company": "STARBUCKS STORE #10208",
  "address": "11302 EUCLID AVENUE, CLEVELAND, OH (216) 229-0749",
  "date": "14/03/2015",
  "total": "4.95"
}
```
