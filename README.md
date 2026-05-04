# 🧠 Brain Tumor Project

A comprehensive Brain Tumor Analysis project submitted for two courses:
- **Social Informatics** — MRI Image Classification using Machine Learning
- **Big Data** — Patient Records Analysis using Big Data tools

---

## 👥 Team Members
- [Mhmd khaled]
- [ABD ELrahman Abo Arab]
- [ABD ELrahman Mhmd]
- [Mhmd Magdy]
---

## 📁 Project Structure
Brain-Tumor-Project/
├── Social_Informatics/
│   ├── brain_tumor_classifier_colab.py
│   └── requirements.txt
│
├── Big_Data/
│   ├── big_data_brain_tumor.py
│   ├── brain_tumor_dataset.csv
│   ├── requirements.txt
│   ├── Dockerfile
│   └── charts/
│       ├── chart1_bar_tumor_type.png
│       ├── chart2_pie_treatment_response.png
│       ├── chart3_line_survival_by_age.png
│       ├── chart4_histogram_tumor_size.png
│       ├── chart5_scatter_size_vs_survival.png
│       └── chart6_heatmap_correlation.png
│
└── README.md





---

## 🔬 Social Informatics Part

### Dataset
- **Type:** MRI Brain Scan Images
- **Classes:** Normal, Glioma Tumor, Meningioma Tumor, Pituitary Tumor
- **Total Images:** 3,096

### Models Used
| Model | Accuracy | AUC |
|-------|----------|-----|
| Random Forest | 77.9% | 0.940 |
| Logistic Regression | 76.3% | 0.908 |
| SVC | 79.5% | 0.952 |
| Decision Tree | 61.9% | 0.747 |
| KNN | 72.4% | 0.917 |
| Ensemble (Soft Vote) | Best | ~0.960+ |

### How to Run
1. Open Google Colab
2. Upload `brain_tumor_classifier_colab.py`
3. Install dependencies:
```python
!pip install -r requirements.txt
```
4. Run the classifier

---

## 📊 Big Data Part

### Dataset
- **Source:** [Kaggle - Brain Tumor Dataset](https://www.kaggle.com/datasets/miadul/brain-tumor-dataset)
- **Type:** Patient Records CSV
- **Size:** 20,000 rows × 20 columns

### Project Parts
| Part | Description |
|------|-------------|
| Part 1 | Data Preprocessing (missing values, duplicates, inconsistencies) |
| Part 2 | Docker Environment Setup |
| Part 3 | Simulated MapReduce (Hadoop Concept) |
| Part 4 | Data Visualization (6 charts) |
| Part 5 | Key Insights & Findings |

### How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the project:
```bash
python big_data_brain_tumor.py
```
3. For Docker:
```bash
docker build -t brain-tumor-bigdata .
docker run -p 8888:8888 brain-tumor-bigdata
```

---

## 🛠️ Technologies Used
- Python 3.13.3
- Scikit-learn
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Docker
- Google Colab
- Jupyter Notebook
