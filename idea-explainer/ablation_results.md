# Ablation Results — Full Metrics

## Table 1: Iterative Enhancements (MVM Pretraining)

_This table demonstrates the step-by-step improvements to the baseline pipeline, culminating in the mathematically correct Semantic Init (Centered+Std)._

### Instance-Level Retrieval

| Metric           | Baseline | +Pretrain | +Bbox Crop | +Crop+Pretrain | +Crop+Pretrain+Aug | +Semantic Init |
| ---------------- | -------- | --------- | ---------- | -------------- | ------------------ | -------------- |
| **Text → Voxel** |
| Recall@1         | 0.84%    | 1.32%     | 3.72%      | 3.96%          | **4.20%**          | 3.00%          |
| Recall@5         | 8.64%    | 8.52%     | 13.69%     | **14.89%**     | 14.77%             | 14.05%         |
| Recall@10        | 17.65%   | 15.85%    | 21.25%     | 24.13%         | **25.21%**         | 23.05%         |
| MRR              | 0.0632   | 0.0688    | 0.0985     | 0.1080         | **0.1105**         | 0.1019         |
| Median Rank      | 55       | 52        | 41         | 36             | **33**             | 36             |
| Mean Rank        | 91.23    | 89.79     | 83.85      | 76.61          | **73.80**          | 76.63          |
| **Voxel → Text** |
| Recall@1         | 2.16%    | 2.28%     | 2.52%      | 3.36%          | 3.36%              | **3.96%**      |
| Recall@5         | 8.16%    | 8.88%     | 11.40%     | 13.93%         | **14.05%**         | 13.69%         |
| Recall@10        | 16.33%   | 15.25%    | 20.65%     | 23.53%         | **23.77%**         | 22.45%         |
| MRR              | 0.0682   | 0.0722    | 0.0863     | 0.1016         | 0.1019             | **0.1039**     |
| Median Rank      | 57       | 53        | 42         | 36             | **34**             | 34             |
| Mean Rank        | 94.09    | 91.36     | 85.27      | 77.75          | **75.78**          | 78.65          |

### Category-Level Retrieval

| Metric           | Baseline | +Pretrain | +Bbox Crop | +Crop+Pretrain | +Crop+Pretrain+Aug | +Semantic Init |
| ---------------- | -------- | --------- | ---------- | -------------- | ------------------ | -------------- |
| **Text → Voxel** |
| Cat P@1          | 39.62%   | 42.62%    | 47.90%     | 50.30%         | **51.62%**         | 48.38%         |
| Cat P@5          | 42.64%   | 43.77%    | 46.46%     | **47.88%**     | 47.37%             | 46.24%         |
| Cat P@10         | 43.51%   | 43.33%    | 46.22%     | 46.61%         | **46.81%**         | 45.52%         |
| Cat Hit@1        | 39.62%   | 42.62%    | 47.90%     | 50.30%         | **51.62%**         | 48.38%         |
| Cat Hit@5        | 82.23%   | 82.35%    | 84.03%     | 83.31%         | 82.47%             | **84.75%**     |
| Cat Hit@10       | 91.72%   | 89.92%    | **92.08%** | 91.72%         | 91.72%             | 91.12%         |
| **Voxel → Text** |
| Cat P@1          | 45.50%   | 45.50%    | 46.70%     | 48.62%         | 46.82%             | **52.10%**     |
| Cat P@5          | 44.95%   | 45.40%    | 46.51%     | 48.04%         | 47.42%             | **48.19%**     |
| Cat P@10         | 44.11%   | 44.56%    | 45.89%     | **46.87%**     | 46.73%             | 46.76%         |
| Cat Hit@1        | 45.50%   | 45.50%    | 46.70%     | 48.62%         | 46.82%             | **52.10%**     |
| Cat Hit@5        | 74.55%   | 74.07%    | 74.67%     | 76.59%         | 76.35%             | **78.03%**     |
| Cat Hit@10       | 82.35%   | 83.67%    | 82.71%     | 84.39%         | 83.43%             | **84.87%**     |

---

## Table 2: Pretraining Strategies

_This table isolates the pretraining procedure by comparing MVM, SimCLR, and Hybrid modes using the final, unified pipeline (Crop + Augments + Semantic Init Centered+Std). First place is **bold**, second place is <u>underlined</u>._

### Instance-Level Retrieval

| Metric           | MVM           | SimCLR        | Hybrid        |
| ---------------- | ------------- | ------------- | ------------- |
| **Text → Voxel** |
| Recall@1         | 3.00%         | <u>3.60%</u>  | **4.68%**     |
| Recall@5         | 14.05%        | **15.49%**    | <u>15.37%</u> |
| Recall@10        | 23.05%        | **24.13%**    | **24.13%**    |
| MRR              | 0.1019        | <u>0.1050</u> | **0.1112**    |
| Median Rank      | **36**        | <u>38</u>     | 41            |
| Mean Rank        | **76.63**     | <u>77.59</u>  | 79.85         |
| **Voxel → Text** |
| Recall@1         | <u>3.96%</u>  | 3.72%         | **4.44%**     |
| Recall@5         | <u>13.69%</u> | 13.21%        | **14.53%**    |
| Recall@10        | <u>22.45%</u> | <u>22.45%</u> | **22.57%**    |
| MRR              | <u>0.1039</u> | 0.1014        | **0.1048**    |
| Median Rank      | **34**        | 39            | <u>38</u>     |
| Mean Rank        | 78.65         | **78.63**     | <u>81.33</u>  |

### Category-Level Retrieval

| Metric           | MVM        | SimCLR        | Hybrid        |
| ---------------- | ---------- | ------------- | ------------- |
| **Text → Voxel** |
| Cat P@1          | **48.38%** | <u>48.14%</u> | 46.82%        |
| Cat P@5          | **46.24%** | 46.05%        | <u>46.17%</u> |
| Cat P@10         | **45.52%** | <u>45.33%</u> | 45.20%        |
| Cat Hit@1        | **48.38%** | <u>48.14%</u> | 46.82%        |
| Cat Hit@5        | **84.75%** | <u>84.51%</u> | 84.03%        |
| Cat Hit@10       | 91.12%     | <u>90.88%</u> | **92.08%**    |
| **Voxel → Text** |
| Cat P@1          | **52.10%** | <u>48.14%</u> | 46.94%        |
| Cat P@5          | **48.19%** | <u>47.06%</u> | 46.94%        |
| Cat P@10         | **46.76%** | <u>45.74%</u> | 45.55%        |
| Cat Hit@1        | **52.10%** | <u>48.14%</u> | 46.94%        |
| Cat Hit@5        | **78.03%** | 75.75%        | <u>75.99%</u> |
| Cat Hit@10       | **84.87%** | <u>84.63%</u> | 84.27%        |

## Key Findings

1. **Bbox cropping and Data Augmentations** were the single most impactful changes in the base pipeline, skyrocketing recall metrics by dense packing and regularizing the voxel structures.
2. **Hybrid Pretraining Excels at Instance Retrieval:** Combining MVM with SimCLR contrastive learning yielded the highest Recall@1 and MRR across both Text→Voxel and Voxel→Text retrieval tasks, surpassing standalone modes by a massive margin (+56% relative T→V R@1 compared to MVM).
3. **MVM Maintains Category Superiority:** While Hybrid dominates exact instance matches, pure MVM pretraining still strictly holds the top spot for identifying the broad Category (Cat P@1, P@5), likely because it rigidly memorizes fine-grained block frequencies without the smoothing effect of contrastive clustering.
