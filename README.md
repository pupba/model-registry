# Model-Registry

mlflow, MinIO, PostgreSQL을 사용해서 Model-Registry Project 구축.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### **시스템 구성**

![Alt text](./img/image.png)

### Skills

-   Docker
-   Docker-Compose
-   Python 3.9.0
-   postgres 14.0
-   minio RELEASE.2024-01-18T22-51-28Z

## 1. 실행환경을 위한 docker-compose, Dockerfile 작성

-   MLflow : `http://localhost:5001`
    ![Alt text](./img/image-1.png)
-   MinIO : `http://localhost:9001`
    ![Alt text](./img/image-2.png)

### 2. Save_model - save_model.py

-   Pytorch 모델을 만들어 학습하고 Lifecycle을 관리하기 위해 mlflow를 사용.
    ![Alt text](./img/image-3.png)
    ![Alt text](./img/image-4.png)
    ![Alt text](./img/image-5.png)

### 3. Load_model - load_model.py

-   mlflow를 사용해서 모델을 불러온다.
    ![Alt text](./img/image-6.png)
