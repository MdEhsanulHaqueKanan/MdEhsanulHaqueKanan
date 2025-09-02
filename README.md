# Hi, I'm Ehsanul. 👋
## End-to-End AI Engineer | LLMs, RAG, NLP, & MLOps

I build intelligent, data-driven applications from prototype to production. My passion lies in taking complex AI concepts—from Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to traditional NLP—and engineering them into deployed, containerized, and scalable applications that solve real-world problems.

🧠 **My specialization is in End-to-End AI & MLOps:** I excel at building and deploying complex AI systems, including RAG pipelines and AI agents, by optimizing for production constraints like cost, privacy, and performance.

🚀 **I deliver production-ready solutions:** I have hands-on experience with the full lifecycle: from model selection and prompt engineering to containerization (Docker), cloud deployment, and local inference with tools like Ollama. I'm adept at working across diverse environments and time zones.

🌐 **Fluent in English:** Duolingo English Test (DET) score of 135/160 (equivalent to IELTS 7.0).


---

## 🛠️ My Core Capabilities

**End-to-End AI Engineering | Large Language Models (LLMs) & RAG | Natural Language Processing (NLP) | Machine Learning Operations (MLOps) | Computer Vision | Data Science & Engineering**

---

## 🔬 Independent Research & Development

This section highlights significant independent projects where I've deep-dived into a complex domain, applying advanced AI/ML and MLOps principles. It demonstrates my ability to conceptualize, research, and execute a comprehensive, industry-standard AI solution from the ground up.

### 🚁 RescueVision: AI-Powered Search & Rescue Command Center

An end-to-end, multi-modal AI system designed to accelerate search and rescue (SAR) operations. This full-stack application functions as a professional-grade **after-action review and intelligence gathering tool**, allowing operators to rapidly analyze hours of aerial drone footage and pinpoint critical events with pixel-perfect accuracy.

The platform is built on a robust microservices architecture, fusing two powerful, custom-built AI capabilities into a single, intuitive interface: a high-performance **Computer Vision survivor detection pipeline** and a **Retrieval-Augmented Generation (RAG) Mission AI Assistant** grounded in official FEMA SAR manuals.

#### **Key Achievements & Performance Metrics:**

*   **Flawless UX for High-Stakes Analysis:** Architected a sophisticated "Event Reviewer" workflow. Operators upload a video, which is processed by the backend. The UI then populates a log of all detected events. Clicking an event jumps the video to the precise timestamp and draws a **static, pixel-perfect bounding box**, eliminating the ambiguity and latency of real-time trackers.
*   **High-Performance Custom CV Model:** Engineered and trained a state-of-the-art **YOLOv8n object detection model** from scratch on a challenging aerial dataset of over **2,200 images**. Achieved exceptional performance metrics on the validation set, proving the model's accuracy in identifying small, difficult-to-detect human figures from above:
    *   **Precision:** **85.3%**
    *   **Recall:** **76.2%**
    *   **mAP50:** **0.833**
*   **Authoritative RAG-Powered AI Assistant:** Built a complete RAG pipeline using **LangChain** and **ChromaDB**. The system is grounded in a knowledge base of official **FEMA Search and Rescue manuals**, providing operators with accurate, reliable answers to complex procedural and safety questions.
*   **End-to-End MLOps & System Design:** Executed the full project lifecycle, from data sourcing and model training on Kaggle GPUs to building two independent **Flask-based AI microservices** and integrating them with a stunning, responsive **React/TypeScript** frontend.

#### **Key Skills Demonstrated:**
End-to-End System Architecture, Computer Vision (YOLOv8, PyTorch, OpenCV), Retrieval-Augmented Generation (RAG), NLP (Sentence-Transformers), Full-Stack Development (React, TypeScript, Flask), MLOps (Data Curation, Model Training, API Development), Strategic Product Pivoting.

#### **Tech Stack:**
*   **AI Backend:** Python, PyTorch, Ultralytics, OpenCV, LangChain, ChromaDB, Flask
*   **Frontend:** React, TypeScript, Vite, Tailwind CSS
*   **AI Research:** Kaggle Notebooks, Pandas, Git LFS

**[📂 Frontend Source Code](https://github.com/MdEhsanulHaqueKanan/RescueVision-Frontend)** | **[📂 AI Backend Source Code](https://github.com/MdEhsanulHaqueKanan/RescueVision-AI-Backend)**

### 🏭 Multimodal Knowledge-Enhanced Prescriptive Maintenance System
An advanced, end-to-end **Prescriptive Maintenance (PdM)** system that demonstrates an intelligent, full-stack MLOps workflow. This project evolves beyond traditional prediction by integrating a **Retrieval-Augmented Generation (RAG)** pipeline, transforming it from a system that predicts *when* an asset will fail to one that prescribes *how to fix it* via a conversational AI assistant.

#### Highlights:
*   **Achieved strong predictive performance** on the underlying models, with an **RMSE of 15.82** for RUL prediction and **up to 95% Recall** for fault classification on imbalanced data.
*   **Built a complete, private RAG pipeline** using **Ollama**, **ChromaDB**, and **LangChain** to create a conversational AI assistant that provides accurate, grounded answers from a custom knowledge base of technical manuals and logs.
*   **Implemented a full Human-in-the-Loop (HITL) framework** with UI feedback buttons and a backend endpoint, designing the system for continuous learning and adaptation.
*   **Engineered a complete MLOps lifecycle**, including data ingestion scripts, model training notebooks, a robust Flask API, and a fully **containerized Docker** environment for reproducible local deployment.

#### Key Skills Demonstrated:
End-to-End MLOps, Retrieval-Augmented Generation (RAG), LLMs (Ollama), Vector Databases (ChromaDB), Conversational AI, Human-in-the-Loop (HITL) Design, Explainable AI (XAI), Time-Series Analysis, NLP (Topic Modeling), Containerization (Docker).

#### Tech Stack:
Python, Flask, LangChain, Ollama, ChromaDB, Sentence-Transformers, scikit-learn, XGBoost, LightGBM, SHAP, NLTK, Pandas, Plotly.js, Pytest, Docker.

**[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/multimodal-prescriptive-pdm)** 

---

## ✨ Full-Stack MLOps Applications

These projects highlight my expertise in building complete, end-to-end AI applications, from a modern, interactive frontend to a robust, scalable backend infrastructure.

### 🚗 AuraScanAI: Vehicle Damage Assessment System (Full-Stack with PyTorch & React)
An advanced, end-to-end computer vision project that demonstrates a full MLOps lifecycle for damage assessment. This project was built from the ground up, starting from a research paper and culminating in a fully deployed, full-stack application. It features a sophisticated AI backend that serves a custom-trained Vision Transformer model and a stunning, data-driven React frontend.

#### Highlights:
*   **Custom-Trained Vision Transformer (ViT):** Engineered the complete training pipeline from scratch, including aggregating and processing over **15,500 images** from multiple datasets. Fine-tuned a `vit_base_patch16_224` model on a Kaggle GPU, achieving a best **validation loss of 248.27**, proving the model's ability to learn complex damage features.
*   **Professional MLOps Workflow:** Utilized **Docker** for containerization and **Git LFS** to professionally manage the large (~343 MB) model file, ensuring a reproducible and scalable deployment.
*   **Advanced Full-Stack Architecture:** Architected and deployed a modular system with a **Flask/PyTorch API containerized on Hugging Face Spaces** and a **React/TypeScript frontend on Vercel**.
*   **Dynamic Frontend Visualization:** The React UI dynamically renders a bounding box over the primary damage area based on raw JSON coordinates received from the live API, demonstrating advanced frontend data visualization.
*   **Business Logic Integration:** The backend includes a "Business Rule Engine" that post-processes the AI's raw output to provide user-friendly severity classifications and realistic, estimated repair cost ranges.

#### Tech Stack:
**Frontend:** React, TypeScript, Vite, Tailwind CSS, Vercel
**Backend:** Python, Flask, PyTorch, `timm`, Docker, Git LFS, Hugging Face Spaces
**AI Research:** Kaggle Notebooks, Pandas, OpenCV

**[➡️ Live Demo](https://aurascan-ai.vercel.app/)** | **[📂 Data Exploration & Model Training Code](https://github.com/MdEhsanulHaqueKanan/aurascan-ai)** | **[📂 Frontend Code](https://github.com/MdEhsanulHaqueKanan/aurascan-frontend)** | **[📂 Backend API Code](https://github.com/MdEhsanulHaqueKanan/aurascan-api)** 

*(Note: The backend API is on a free community tier and may "sleep" after inactivity. The first analysis might take 30-60 seconds to wake the server up.)*

---

### ☁️ AWS Serverless + React Full-Stack YouTube Popularity Predictor 
A complete, end-to-end MLOps project that pairs a powerful serverless backend on AWS with a stunning, modern React and TypeScript frontend. This project showcases the ability to build and integrate a full-stack system, delivering a seamless user experience from data input to ML-powered prediction.

#### Highlights:
*   **Engineered an End-to-End Prediction Pipeline:** Handled the entire ML workflow from scratch, including feature engineering on a complex real-world dataset, training multiple regression models to achieve an **R² of 0.63**, and serializing the final model for production.
*   **Engineered a zero-cost prediction infrastructure** using the **AWS Always Free Tier** (Lambda, API Gateway, SЗ), demonstrating cost-conscious cloud architecture.
*   **Built a full CI/CD pipeline for the backend** using **AWS CodeBuild and GitHub** to automatically containerize and deploy the prediction model.
*   **Architected a fully decoupled, full-stack system**, with a React/TypeScript frontend communicating with a serverless Python backend via a REST API.
*   **Developed a stunning, intuitive user interface** using React and Tailwind CSS, bootstrapped with Google AI Studio, to create a premium user experience for the prediction service.

#### Tech Stack:
**Frontend:** React, TypeScript, Vite, Tailwind CSS
**Backend:** AWS (Lambda, API Gateway, CodeBuild, ECS Fargate, S3), Docker, Python, Scikit-Learn

**[➡️ Live Demo](http://youtube-predictor-frontend-ehsanul-72525.s3-website-us-east-1.amazonaws.com/)** | **[📂 Frontend Code](https://github.com/MdEhsanulHaqueKanan/youtube-predictor-frontend)** | **[📂 Backend Code](https://github.com/MdEhsanulHaqueKanan/aws-serverless-youtube-predictor)**

---

### 🤖 NLP-Powered Disease Diagnosis System (Full-Stack with React & Flask)
An end-to-end NLP application that has been evolved into a decoupled, full-stack system. This project demonstrates the complete lifecycle: from a monolithic proof-of-concept to a scalable, production-grade application with a modern React UI and a containerized Flask API.

#### Highlights:
*   **Production-Optimized Backend:** Optimized the core NLP model's memory footprint by **~75% (to 22MB)** to successfully deploy on a **512MB RAM** server, achieving sub-second API response times on cost-free infrastructure.
*   **Full-Stack Decoupled Architecture:** Architected and deployed a modular system with a **Flask API containerized on Render** and a **React/TypeScript frontend on Vercel**.
*   **Modern React UI:** Built a stunning, responsive user interface with React and TypeScript, bootstrapped with **Google AI Studio**, featuring data visualizations and animations for a premium user experience.


#### Tech Stack:
React, TypeScript, Vite, Python, Flask, Sentence-Transformers, Docker, Render (API), Vercel (Frontend).

**[➡️ Live Demo](https://symptom-checker-frontend-ten.vercel.app/)** | **[📂 Frontend Code](https://github.com/MdEhsanulHaqueKanan/symptom-checker-frontend)** | **[📂 Backend API Code](https://github.com/MdEhsanulHaqueKanan/disease-diagnosis-api)** | **[📂 Original Project](https://github.com/MdEhsanulHaqueKanan/disease-diagnosis-system)**

*(Note: The backend API on the free tier may spin down after inactivity. Please allow up to a minute for the app to "wake up" on your first visit.)*

---

### 🧠 Processor Recommendation Engine (Full-Stack with React & Flask)
A complete, end-to-end AI application that combines a high-accuracy machine learning backend with a modern, interactive React frontend. This project demonstrates the full product lifecycle: from data engineering and model training to building a headless API and deploying a decoupled, full-stack system to the cloud.

#### Key Achievements & Results:
*   **Custom-Trained Classification Model (92.7% Accuracy):** Engineered the complete pipeline from data engineering and feature selection to training and hyperparameter tuning a multi-class classification model, lifting final performance by **2.6%**.
*   **Full-Stack Decoupled Architecture:** Architected and deployed a modular system with a **Flask API containerized on Render** and a **React/TypeScript frontend on Vercel**.

#### Tech Stack:
React, TypeScript, Vite, Python, Flask, Pandas, Scikit-Learn, Docker, Render (API Hosting), Vercel (Frontend Hosting).

**[➡️ Live Demo](https://processor-analysis-frontend.vercel.app/)** | **[📂 Frontend Source Code](https://github.com/MdEhsanulHaqueKanan/processor-analysis-frontend)** | **[📂 Backend API Source Code](https://github.com/MdEhsanulHaqueKanan/processor-recommendation-api)** **[📂 Original Project](https://github.com/MdEhsanulHaqueKanan/processor-recommendation-engine)** 

*(Note: The backend server on the free tier may spin down after inactivity. Please allow up to a minute for the app to "wake up" on your first visit.)*

---

### ✈️ Drone Detection System (Full-Stack with PyTorch & React)
An advanced, end-to-end computer vision project that demonstrates a full MLOps lifecycle for object detection. This project was evolved from a monolithic Flask application into a modern, decoupled system featuring a containerized PyTorch API that returns JSON coordinates, and a dynamic React frontend that renders bounding boxes on the client-side.

#### Highlights:
*   **Custom-Trained Object Detection Model:** Engineered the complete training pipeline from scratch, including data preprocessing and writing the PyTorch training and validation loops. Fine-tuned a **Faster R-CNN** model with PyTorch on a Kaggle GPU, achieving a low final test loss of **0.0638**, indicating strong model accuracy and convergence.
*   **Professional MLOps Workflow:** Utilized **Docker** for containerization and **Git LFS** to professionally manage large model files, ensuring a reproducible and scalable deployment.
*   **Advanced Full-Stack Architecture:** Architected and deployed a modular system with a **Flask/PyTorch API containerized on Hugging Face Spaces** and a **React/TypeScript frontend on Vercel**.
*   **Complex Frontend Rendering:** The React UI dynamically renders bounding boxes and labels over the source image based on raw JSON coordinates received from the API, demonstrating advanced frontend data visualization.

#### Tech Stack:
**Frontend:** React, TypeScript, Vite, Tailwind CSS, Vercel
**Backend:** Python, Flask, PyTorch, Docker, Git LFS, Hugging Face Spaces

**[➡️ Live Demo](https://drone-detection-frontend.vercel.app/)** | **[📂 Frontend Code](https://github.com/MdEhsanulHaqueKanan/drone-detection-frontend)** | **[📂 Backend API Code](https://github.com/MdEhsanulHaqueKanan/drone-detection-api)** | **[📂 Original Project](https://github.com/MdEhsanulHaqueKanan/drone-detection-deep-learning-flask-app)**

*(Note: The backend API is on a free community tier and may "sleep" after inactivity. The first prediction might take 30-90 seconds to wake the server up.)*

---

### 🌿 House Plant Species Identifier (Full-Stack with PyTorch & React)
An end-to-end computer vision project that was evolved from a monolithic Flask app into a modern, decoupled, full-stack system. This project demonstrates the complete MLOps lifecycle: from a proof-of-concept to a scalable, production-grade application with a stunning React UI and a containerized PyTorch API deployed on a specialized ML platform.

#### Highlights:
*   **Custom-Trained Deep Learning Model (82% Accuracy):** Engineered the complete training pipeline from scratch, including data augmentation and writing the full PyTorch training and validation loops. Fine-tuned an `EfficientNet-B0` model on a Kaggle GPU to achieve an 82% weighted F1-score on 47 different plant species.
*   **Full-Stack Decoupled Architecture:** Architected and deployed a modular system with a **Flask/PyTorch API containerized on Hugging Face Spaces** and a **React/TypeScript frontend on Vercel**.
*   **Production Platform Migration:** Successfully diagnosed and resolved platform-specific timeout issues by migrating the backend from a general-purpose host (Render) to a specialized, hardware-accelerated ML host (Hugging Face), demonstrating advanced MLOps problem-solving.
*   **Modern React UI:** Built a beautiful, responsive user interface with React and TypeScript, bootstrapped with **Google AI Studio**, to create a premium user experience.

#### Tech Stack:
**Frontend:** React, TypeScript, Vite, Tailwind CSS, Vercel
**Backend:** Python, Flask, PyTorch, Docker, Git LFS, Hugging Face Spaces

**[➡️ Live Demo](https://house-plant-frontend-3vsr32tzq-md-ehsanul-haque-kanans-projects.vercel.app/)** | **[📂 Frontend Code](https://github.com/MdEhsanulHaqueKanan/house-plant-frontend)** | **[📂 Backend API Code](https://github.com/MdEhsanulHaqueKanan/house-plant-api)** | **[📂 Original Project](https://github.com/MdEhsanulHaqueKanan/house-plant-species-identifier-machine-learning-flask-app)**

*(Note: The backend API is on a free community tier and may "sleep" after inactivity. The first prediction might take 30-90 seconds to wake the server up.)*

---

## 🚀 MLOps Showcase

This section highlights projects focused on the core automation and infrastructure of Machine Learning Operations.

###  Scalable ML Service: Ad Sales Prediction with CI/CD & MLflow (Full-Stack with React & Flask)
A complete MLOps pipeline and full-stack application that automatically tests and deploys a containerized Flask API to Render, consumed by a modern React frontend deployed on Vercel.

#### Highlights:
*   **Full CI/CD Pipeline with GitHub Actions:** Automatically runs `pytest` unit tests, builds a Docker image, and pushes the validated image to Docker Hub on every commit to `main`.
*   **Experiment Tracking & Model Registry with MLflow:** Tracks all training runs, logs parameters and metrics for reproducibility, and manages model versions by promoting them to "Production".

#### Tech Stack:
MLflow, GitHub Actions, Pytest, Docker, Docker Hub, Flask, Gunicorn, Scikit-learn, Pandas, Render, React, TypeScript

**[➡️ Live Demo](https://ad-sales-predictor-frontend.vercel.app/)** | **[📂 Frontend Source](https://github.com/MdEhsanulHaqueKanan/ad-sales-predictor-frontend)** | **[📂 Backend Source](https://github.com/MdEhsanulHaqueKanan/Scalable-ML-Service-Ad-Sales-Prediction-with-CI-CD-and-MLflow)** | **[➡️ Live CI/CD Pipeline](https://github.com/MdEhsanulHaqueKanan/Scalable-ML-Service-Ad-Sales-Prediction-with-CI-CD-and-MLflow/actions)** | **[📦 Docker Hub Image](https://hub.docker.com/r/mdehsanulhaquekanan/scalable-ml-service)**

---

## 🔬 Additional End-to-End Projects

A collection of other projects demonstrating a wide range of skills across the AI/ML landscape.

### 🏭 Robust & Interpretable Predictive Maintenance System (PdM)
A comprehensive, end-to-end Predictive Maintenance (PdM) system designed for evolving industrial environments. This project showcases an industry-standard approach to forecasting equipment failures, explaining model decisions, and adapting to changing operational conditions. It embodies the full Machine Learning Operations (MLOps) lifecycle, from data processing and model training to API development, dynamic web visualization, automated testing, and containerized deployment.

#### Highlights:
* Achieved **RMSE 15.82 (R² 0.85)** and **95% Recall with XAI**, significantly reducing dashboard load time from **5 mins to under 5 seconds**.
* Implemented a full **MLOps lifecycle** including **automated `pytest`** and **Docker** containerization for robust, production-ready deployment.

#### Key Skills Demonstrated:
End-to-End MLOps Lifecycle, Machine Learning Research, Data Engineering, Model Development & Deployment, Explainable AI (XAI), Time-Series Analysis, Containerization, Automated Testing.

#### Tech Stack:
Python, Flask, scikit-learn, XGBoost, LightGBM, SHAP, Imbalanced-learn, Pandas, Plotly.js, Pico.css, Pytest, Docker.

**[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/robust-pdm-system)**

---

### 🧠 AI Research Assistant (with Local LLM & RAG)
A full-stack, end-to-end AI agent that answers complex questions with up-to-date, sourced information. This project demonstrates a deep understanding of modern AI agent architecture, running a 100% private and cost-free stack with a local LLM (Phi-3) and a robust, orchestrated RAG workflow.

#### Highlights:
* Re-architected the system to achieve a **100% task success rate** and **67% faster response** for complex queries.
* Reduced required LLM calls per query by **over 80%**, leading to **100% cost reduction** via local-first inference.
#### Tech Stack:
Python, LangChain, Streamlit, Ollama, Phi-3, DuckDuckGo Search.

**[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/ai-research-assistant)** 

*(Note: This project runs locally to ensure 100% data privacy and has no live deployment.)*

---

### 🌦️ Seattle Weather Classifier: A Time-Series ML Application
An end-to-end project that predicts daily weather conditions in Seattle. This showcases a complete ML workflow, from rigorous feature engineering on time-series data to deployment as an interactive web app.

#### Highlights:
* Advanced time-series feature engineering, detailed model comparison, and a fully containerized deployment.
#### Tech Stack:
Python, Flask, Scikit-Learn, Pandas, Docker, Render.

**[➡️ Live Demo](https://flask-ml-weather-prediction.onrender.com/)** | **[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/weather-prediction-machine-learning-flask-app)**

---

### 🔧 Predictive Maintenance Classifier

A full-stack machine learning application designed to predict equipment failure before it occurs. This project showcases a complete MLOps workflow, from data analysis and feature engineering on an imbalanced dataset to deployment as an interactive Flask web app.

#### Highlights:
* Achieved **93% overall accuracy** by training a Random Forest model on a highly imbalanced, real-world dataset.
* Successfully addressed the core challenge of rare failure events to reach a **Macro F1-Score of 0.41**, demonstrating a robust and practical approach to real-world classification.
#### Tech Stack:
Python, Flask, Scikit-Learn, Pandas, Imbalanced-learn, Joblib.

**[📂 Source Code](https://github.com/MdEhsanulHaqueKanan/predictive-maintenance-machine-learning-flask-app)**

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
      <td valign="middle"><strong>Frontend & UI</strong></td>
      <td valign="middle">
        <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=react,vite,ts" /></a>
        <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS" />
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
        <img src="https://img.shields.io/badge/MLflow-0A9BDB?style=for-the-badge&logo=mlflow" alt="MLflow" />
        <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel" />
      </td>
    </tr>
    <tr>
      <td valign="middle"><strong>Cloud Platforms</strong></td>
      <td valign="middle">
        <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render" />
        <img src="https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS" /> (Lambda, API Gateway, CodeBuild, ECS Fargate, S3, ECR, IAM, CloudWatch)
      </td>
    </tr>
    <tr>
      <td valign="middle"><strong>Remote Collaboration</strong></td>
      <td valign="middle">
        <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack" />
        <img src="https://img.shields.io/badge/Zoom-2D8CFF?style=for-the-badge&logo=zoom&logoColor=white" alt="Zoom" />
        <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=git" /></a>
      </td>
    </tr>
  </tbody>
</table>

---

## 🏅 Certifications

* **[IBM's Deep Learning Professional Certificate](https://www.credly.com/badges/854566cd-5688-42c4-9fc7-68a5e06afa07/linked_in_profile)**
* **[IBM's Python Data Science Professional Certificate](https://credentials.edx.org/credentials/e7d354814d4c424ca1a775b5b87c4deb/)**

---

### 📫 Let's Connect!

I'm always open to discussing new projects, creative ideas, or opportunities to be part of your vision.

<p align="left">
  <a href="https://www.linkedin.com/in/ehsanulhaquekanan/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

[![My Website](https://img.shields.io/badge/My_Website-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://ehsanul-ai-engineer.vercel.app)
