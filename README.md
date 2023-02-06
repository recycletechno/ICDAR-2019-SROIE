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

A pretrained model [unstructuredio/donut-base-sroie](https://huggingface.co/unstructuredio/donut-base-sroie) was taken as a backbone

## Results

Mean Levenstein ratio on Task 1,2: **0.9337**

Mean Levenstein ratio on Task 3:   **0.9177**

I also calculated [Intersection over Union index](https://en.wikipedia.org/wiki/Jaccard_index) for text localization task but got avarage results - **0.8274**. When I went into details of index calculation I realized that all the ground truths bounding boxes were rectangles without any slope. But PaddleOCR produce rectangles with slope when image is rotated. Above is an example of rotated image `X51005268408.jpg` from test dataset

![X51005268408.jpg](img/X51005268408_part.png)

Red boxes are predictions and green boxes are the ground truths. IoU ratio is using only 2 vertices (upper left and lower right):

```
pred_box_df = pd.DataFrame(data=lines, columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'text'])

pred_box_df['x'] = pred_box_df['x1'].astype(int)
pred_box_df['y'] = pred_box_df['y1'].astype(int)
pred_box_df['w'] = pred_box_df['x2'].astype(int) - pred_box_df['x1'].astype(int)
pred_box_df['h'] = pred_box_df['y3'].astype(int) - pred_box_df['y1'].astype(int)

pred_box_df = pred_box_df.drop(columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

ap, mean_iou, mean_ratio = calculate_AP(gt_box_df, pred_box_df, iou_thr, lratio_thr)
```

Tail from ground truth `X51005268408.txt` file:

<pre>
196,<b>1335</b>,600,<b>1335</b>,600,<b>1390</b>,196,<b>1390</b>,THANK YOU. PLEASE COME AGAIN
<b>123</b>,1375,<b>683</b>,1375,<b>683</b>,1441,<b>123</b>,1441,KEEP THE INVOICE FOR APPLICABLE RETURNS
</pre>

One can observe that Xs and Ys repeat each other, which leads to zero slope of rectangle

On the other hand PaddleOCR yields angled rectangles with different eight coordinates. And it gains with very good OCR results. That's why I did not rely on IoU ratio too much

<pre>
198,1358,597,1331,599,1365,200,1392,IHANK VOU. PLEASE CONE AOAIN
122,1407,682,1365,685,1397,125,1440,KEEG THE INVOICE FAR SPALICEBLE RETURNS
</pre>

Enlarged lower section:

![X51005268408.jpg](img/X51005268408_part_2.png)

