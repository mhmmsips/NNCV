# Final Assignment: Cityscape Challenge  

Welcome to the **Cityscape Challenge**, the final project for this course!  

In this assignment, you'll put your knowledge of Neural Networks (NNs) for computer vision into action by tackling real-world problems using the **CityScapes dataset**. This dataset contains large-scale, high-quality images of urban environments, making it perfect for tasks like **semantic segmentation** and **object detection**.  

This challenge is designed to push your skills further, focusing on practical and often under-explored issues crucial for deploying computer vision models in real-world scenarios.  

---

## Benchmarks  

The competition comprises four benchmarks, each targeting a specific aspect of model performance:  

1. **Peak performance**  
   This benchmark serves as your baseline. It evaluates the model's segmentation accuracy on a clean, standard test set. Your goal is to achieve the highest segmentation scores here.  

2. **Robustness**  
   This benchmark tests how well your model performs under challenging conditions, such as changes in lighting, weather, or image quality. Consistency is key in this category.  

3. **Efficiency**  
   Practical applications often require compact models. This benchmark emphasizes creating smaller models that maintain acceptable performance. It’s particularly relevant for edge devices where large models are infeasible.  

4. **Out-of-distribution detection**  
   Models often encounter data that differs from the training distribution, leading to unreliable predictions. This benchmark evaluates your model's ability to detect and handle such out-of-distribution samples.  

> **IMPORTANT NOTE**: There will be a fifth benchmark on the competition server where everyone **must** submit a baseline model. The code for training this model is already provided. This benchmark aims to ensure everyone is familiar with working on an HPC cluster. The benchmark will close after **Sunday, March 16**, so start preparing your baseline submission early. This will give you time to ask questions during the arranged computer classes if you encounter any issues.
---

## Deliverables  

Your final submission will consist of the following:  

### 1. Research paper  
Write a **4-page research paper** in IEEE double-column format, addressing the following:  

- **Introduction**: Present the problem, challenges, and potential solutions based on existing literature.  
- **Baseline implementation**: Describe your baseline approach and results using an off-the-shelf segmentation model.  
- **Improvements**: Outline the enhancements you made, supported by experimental results and justifications.  
- **Discussion**: Discuss limitations and suggest future improvements.  
- **Figures & tables**: Use clear visuals to support your findings.

> **Submission**: Submit your paper as a PDF document via **Canvas**.

The paper will be graded based on clarity, experimental design, insight, and originality.  

### 2. Code repository  
Push all relevant code to a **public GitHub repository** with a README.md file detailing:  
- Required libraries and installation instructions.  
- Steps to run your code.  
- Your Codalab username and TU/e email address for correct mapping across systems.  

### 3. Codalab submissions  
Submit your model for evaluation to the **Codalab challenge server**, which includes four benchmark test sets.  

---

## Grading and Bonus Points  

The final assignment accounts for **50% of your course grade**. Additionally, bonus points are available:  

- **Top 3 in any benchmark**: +0.25 to your final assignment grade.  
- **Best performance in any benchmark**: +0.5 to your final assignment grade.  

For example, achieving the best performance in 'Peak Performance' and a top 3 spot in another benchmark will earn you a 0.75 bonus.  

> **Note**: The bonus is optional. A great report with an innovative solution that doesn't rank highly can still earn a perfect score (10).  

---

## Important Notes  

- Ensure a proper **train-validation split** of the CityScapes dataset.  
- Training your model may take many hours; plan accordingly.  
- Use ideas from literature but remember to **cite all sources**. Plagiarism will not be tolerated.  
- For questions or challenges, use the **Discussions** section of this repository to collaborate with peers.  

---

We wish you the best of luck in this challenge and are excited to see the innovative solutions you develop! 🚀