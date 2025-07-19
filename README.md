# Hi, I'm Ehsanul. 👋

## End-to-End AI Engineer | LLMs, RAG, NLP, & MLOps

I build intelligent, data-driven applications from prototype to production. My passion lies in taking complex AI concepts—from Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to traditional NLP—and engineering them into deployed, containerized, and scalable applications that solve real-world problems.

- 🧠 **My specialization is in End-to-End AI & MLOps:** I excel at building and deploying complex AI systems, including RAG pipelines and AI agents, by optimizing for production constraints like cost, privacy, and performance.
- 🚀 **I deliver production-ready solutions:** I have hands-on experience with the full lifecycle: from model selection and prompt engineering to containerization (Docker), cloud deployment, and local inference with tools like Ollama.

---

## 🔬 Independent Research & Development

This section highlights a significant independent project where I've deep-dived into a complex domain, applying advanced AI/ML and MLOps principles from research to deployment. It demonstrates my ability to conceptualize, research, and execute a comprehensive, industry-standard AI solution from the ground up.

### 🏭 Robust & Interpretable Predictive Maintenance System (PdM)
A comprehensive, end-to-end Predictive Maintenance (PdM) system designed for evolving industrial environments. This project showcases an industry-standard approach to forecasting equipment failures, explaining model decisions, and adapting to changing operational conditions. It embodies the full Machine Learning Operations (MLOps) lifecycle, from data processing and model training to API development, dynamic web visualization, automated testing, and containerized deployment.

#### **Highlights:**
* Achieved **Root Mean Squared Error (RMSE) of 15.82 cycles (R² 0.85)** for Remaining Useful Life predictions and **up to 95% Recall for critical fault types** on highly imbalanced datasets.
* Reduced dashboard load time from **over 5 minutes to under 5 seconds** by implementing intelligent, on-demand XAI generation.
* Ensured API and model reliability through a comprehensive suite of **automated `pytest` unit tests**, maintaining robust system behavior.
* Successfully containerized the entire multi-service application with Docker, enabling seamless and consistent deployment across diverse environments.
* Processed **over 20,000 time-series RUL records** and **10,000 classification records**.

#### Key Skills Demonstrated: 
End-to-End MLOps Lifecycle, Machine Learning Research, Data Engineering, Model Development & Deployment, Explainable AI (XAI), Time-Series Analysis, Containerization, Automated Testing.

#### Tech Stack: 
Python, Flask, scikit-learn, XGBoost, LightGBM, SHAP, Imbalanced-learn, Pandas, Plotly.js, Pico.css, Pytest, Docker.
* **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/robust-pdm-system)**

---

### ✨ Featured Projects

Here are some of the projects I'm most proud of. They demonstrate my ability to handle the entire AI/ML lifecycle, from a Jupyter Notebook to a live, user-friendly application.

### 🧠 AI Research Assistant (with Local LLM & RAG)
A full-stack, end-to-end AI agent that answers complex questions with up-to-date, sourced information. This project demonstrates a deep understanding of modern AI agent architecture, running a 100% private and cost-free stack with a local LLM (Phi-3) and a robust, orchestrated RAG workflow.

*   **Highlights:** Re-architected the system to achieve a **100% task success rate on complex queries (from 0%)**. Reduced required LLM calls per query by **over 80% (from 5+ to just 1)**, leading to a **100% cost reduction** via local-first inference. Optimized performance, achieving a **67% reduction in response time** (from ~6 mins to ~2 mins) with a smaller model.
*   **Tech Stack:** Python, LangChain, Streamlit, Ollama, Phi-3, DuckDuckGo Search.
*   **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/ai-research-assistant)** *(Note: This project runs locally to ensure 100% data privacy and has no live deployment.)*

### 🤖 NLP Disease Diagnosis System
A full-stack NLP application that assists users in identifying potential diseases from natural language symptoms. This project showcases advanced MLOps skills, including model optimization for production environments and debugging live deployment issues.

*   **Highlights:** Reduced core model's memory footprint by **~75% (from 90MB to 22MB)** to enable successful deployment on a resource-constrained **512MB RAM** cloud server. Engineered for **100% cost-free infrastructure** (Render's free tier). Achieved **sub-second API response times** for real-time predictions by pre-loading model artifacts. Designed a multi-stage `Dockerfile` for lightweight, portable deployment.
*   **Tech Stack:** Python, Flask, Sentence-Transformers, Scikit-Learn, Docker, Gunicorn, Render.
*   **[➡️ Live Demo](https://disease-diagnosis-system.onrender.com/)** | **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/disease-diagnosis-system)**

*(Note: The server may spin down after inactivity. Please allow 3-6 mins for the app to "wake up" on your first visit.)*

---

### 🧠 Processor Recommendation & Analysis Engine
An end-to-end machine learning application that combines rule-based filtering and ML-powered predictions to assist in processor selection for smart devices. This project highlights a full ML lifecycle, from data engineering and model training to robust API development, containerization, and cloud deployment, solving real-world production challenges.

* **Highlights:** Developed a multi-class classification model achieving **92.7% accuracy** (lifted by **2.6%** through hyperparameter tuning). Engineered **9 new, structured features** from raw text for enhanced model performance. Architected and deployed a modular Flask application serving **2 distinct ML models** via **2 primary API endpoints**, containerized with Docker. Successfully processed **over 1,000 unique processors**, demonstrating scalable data handling.
* **Tech Stack:** Python, Flask, Scikit-Learn, Pandas, Docker, Gunicorn, WhiteNoise, Render.
* **[➡️ Live Demo](https://processor-recommendation-engine.onrender.com/)** | **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/processor-recommendation-engine)**

*(Note: The server may spin down after inactivity. Please allow 3-6 mins for the app to "wake up" on your first visit.)*

---

### 🌦️ Seattle Weather Classifier: A Time-Series ML Application
An end-to-end project that predicts daily weather conditions in Seattle. This showcases a complete ML workflow, from rigorous feature engineering on time-series data to deployment as an interactive web app.

*   **Highlights:** Advanced time-series feature engineering, detailed model comparison, and a fully containerized deployment.
*   **Tech Stack:** Python, Flask, Scikit-Learn, Pandas, Docker, Render.
*   **[➡️ Live Demo](https://flask-ml-weather-prediction.onrender.com/)** | **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/weather-prediction-machine-learning-flask-app)**

---

### 🔧 Predictive Maintenance Classifier

A full-stack machine learning application designed to predict equipment failure before it occurs. This project showcases a complete MLOps workflow, from data analysis and feature engineering on an imbalanced dataset to deployment as an interactive Flask web app.

*   **Highlights:** Achieved **93% overall accuracy** by training a Random Forest model on a highly imbalanced, real-world dataset. Successfully addressed the core challenge of rare failure events to reach a **Macro F1-Score of 0.41**, demonstrating a robust and practical approach to real-world classification.
*   **Tech Stack:** Python, Flask, Scikit-Learn, Pandas, Imbalanced-learn, Joblib.
*   **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/predictive-maintenance-machine-learning-flask-app)**

---

### 🌿 House Plant Species Identifier
A deep learning-powered web app that can identify different species of house plants from an uploaded image.

*   **Highlights:** Achieved **82% overall accuracy** (with a weighted F1-score of 0.82) on 47 plant species using a pre-trained computer vision model for transfer learning, deployed in a user-friendly Flask application.. 
*   **Tech Stack:** Python, Flask, PyTorch, Torchvision, Pillow.
*   **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/house-plant-species-identifier-machine-learning-flask-app)**

---

### ✈️ Drone Detection System
An end-to-end computer vision system that trains a Faster R-CNN model for object detection and deploys it as an interactive Flask web application. This project demonstrates a complete workflow from a research-oriented Kaggle notebook to a user-facing, production-ready application.

*   **Highlights:** Achieved a low final test loss of **0.0638**, indicating strong model convergence and accuracy. Successfully transitioned a complex deep learning pipeline from a Kaggle research notebook to a fully interactive Flask app. Engineered a custom visualization pipeline using **Pillow** for clear, professional-grade bounding box and label rendering, significantly improving on default library outputs.
*   **Tech Stack:** Python, Flask, PyTorch, Torchvision, Albumentations, Pillow, NumPy.
*   **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/drone-detection-deep-learning-flask-app)**

---

### 🛠️ My Technology Stack

My toolkit is built on a foundation of robust, industry-standard technologies to deliver high-performance applications.

<table>
  <tbody>
    <tr>
      <td width="150px" valign="middle"><strong>AI & LLM Tooling</strong></td>
      <td width="800px" valign="middle">
        <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=pytorch,tensorflow,scikitlearn" /></a>
        <img src="https://img.shields.io/badge/LangChain-white?style=for-the-badge&logo=langchain" alt="LangChain" />
        <img src="https://img.shields.io/badge/Ollama-grey?style=for-the-badge&logo=ollama" alt="Ollama" />
      </td>
    </tr>
    <tr>
      <td valign="middle"><strong>Backend & UI</strong></td>
      <td valign="middle">
        <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=python,flask,gunicorn" /></a>
        <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
      </td>
    </tr>
    <tr>
      <td valign="middle"><strong>Data Science</strong></td>
      <td valign="middle">
        <img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
        <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
        <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
      </td>
    </tr>
    <tr>
      <td valign="middle"><strong>MLOps & Deployment</strong></td>
      <td valign="middle">
        <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=docker,git,github,githubactions" /></a>
      </td>
    </tr>
  </tbody>
</table>

---

### 📫 Let's Connect!

I'm always open to discussing new projects, creative ideas, or opportunities to be part of your vision.

<p align="left">
  <a href="https://www.linkedin.com/in/ehsanulhaquekanan/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>
