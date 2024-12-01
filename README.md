# Image Classification: Fruits vs. Vegetables using CNN, TensorFlow, Keras, and Streamlit

This project implements an **image classification model** to distinguish between fruits and vegetables. The model is trained using a Convolutional Neural Network (CNN) built with **TensorFlow** and **Keras**, and the results are deployed with an interactive interface using **Streamlit**.

## 📁 Dataset
- **Categories**:
  - Fruits
  - Vegetables
- Contains labeled images of various fruits and vegetables used for training and testing the model.

## 🔧 Project Workflow
1. **Data Preprocessing**:
   - Images resized for uniformity.
   - Normalization applied for efficient training.
2. **Model Architecture**:
   - A CNN model designed with multiple convolutional and pooling layers.
   - Optimized with an Adam optimizer for better performance.
3. **Model Training**:
   - Model trained on the dataset with categorical cross-entropy as the loss function.
   - Achieved significant accuracy in classifying images.
4. **Deployment**:
   - Integrated with Streamlit for an easy-to-use web interface.
   - Users can upload an image to get real-time predictions.

## 🛠️ Tools & Technologies Used
- **TensorFlow** and **Keras**: For building and training the CNN.
- **Streamlit**: For creating an interactive front-end.
- **Python Libraries**:
  - NumPy and Pandas for data handling.
  - Matplotlib for visualizing training results.

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/ramnaresh-ahi/Image-Classifier.git
