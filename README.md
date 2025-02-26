# Customer Segmentation with PyCaret

## 📌 Project Overview
This project focuses on customer segmentation using **unsupervised learning** techniques, specifically **K-Means clustering** with **PyCaret**. The goal is to identify different customer groups based on their age, income, spending behavior, and savings.

A **Streamlit web app** is included to allow users to predict the customer segment of new data points either **online (manual input)** or via **batch processing (CSV upload)**.

---

## 📁 Project Structure
```
PyCaret Project/
│── app1.py
│── Final Kmeans Model.pkl
│── jewel_data.csv
│── logs.log
│── notebook.ipynb
│── test.csv
│── customer_segmentation.png
```

### **Main Files**
- **`notebook.ipynb`** → Contains the full **PyCaret clustering pipeline** with model training, visualization, and evaluation.
- **`app1.py`** → A **Streamlit application** for customer segmentation prediction.
- **`Final Kmeans Model.pkl`** → The trained **K-Means clustering model**.
- **`jewel_data.csv`** → The dataset used for training the clustering model.
- **`customer_segmentation.png`** → Cluster visualization.
- **`logs.log`** → Log file capturing execution details.

---

## 🚀 Installation & Setup
### **1️⃣ Clone the repository**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```
Ensure you have `pycaret`, `streamlit`, `pandas`, `numpy`, `PIL` installed.

### **3️⃣ Run the Streamlit App**
```bash
streamlit run app1.py
```

---

## 🧠 Clustering Model
The **K-Means model** segments customers into 4 distinct groups based on their **Age, Income, Spending Score, and Savings**.

```python
from pycaret.clustering import *
exp1 = setup(data=jewel_data, normalize=True)
kmeans = create_model('kmeans', num_clusters=4)
```

### **Assigning Clusters to Data**
```python
from pycaret.clustering import assign_model
clustered_data = assign_model(kmeans)
print(clustered_data.head())
```

### **Cluster Summary**
```python
cluster_info = clustered_data.groupby('Cluster').mean()
print(cluster_info)
```

---

## 📊 Visualizing Clusters

### **3D Cluster Plot**
```python
plot_model(kmeans, plot='cluster')
```

### **2D PCA Cluster Plot**
```python
plot_model(kmeans, plot='pca')
```

---

## 🎯 Streamlit Web App Features
- **Online Prediction Mode** → Input values manually and predict the customer segment.
- **Batch Prediction Mode** → Upload a CSV file and predict segments for multiple customers at once.

### **How to Use the App?**
- Enter values for **Age, Income, Spending Score, and Savings**.
- Click on **Predict** to see the assigned cluster.

---

## 📌 Example Predictions
| Cluster  | Age | Income | Spending Score | Savings | Interpretation |
|----------|------|--------|----------------|---------|---------------|
| **0** | 59.96 | 72,332 | 0.77 | 6,890 | Older customers with moderate income and high spending |
| **1** | 35.42 | 105,228 | 0.30 | 14,937 | Younger professionals with high income and savings |
| **2** | 87.77 | 27,866 | 0.32 | 16,659 | Retirees with low income but high savings |
| **3** | 40.87 | 126,031 | 0.65 | 7,287 | High-income individuals with moderate spending |

