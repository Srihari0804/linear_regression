# Linear Regression: Scikit-Learn vs. From Scratch

## 📌 Overview

This repository explores the mechanics of **Linear Regression** by implementing it in two distinct ways: using the industry-standard `scikit-learn` library and building it entirely from scratch using `NumPy`. This side-by-side comparison demonstrates both the practical application of machine learning tools and the foundational calculus and linear algebra that power them.

This project was developed as part of the *Calculus* course within the *Mathematics for Machine Learning* specialization offered by DeepLearning.AI.

## ✨ Features

* **From-Scratch Implementation:** Builds the gradient descent algorithm and cost function calculation using only NumPy, showcasing the underlying mathematics.
* **Scikit-Learn Implementation:** Utilizes standard machine learning libraries to demonstrate how regression is handled in production environments.
* **Interactive Comparison:** All code, visualizations, and comparative analyses are contained within a single, easy-to-follow Jupyter Notebook.

## 🧮 The Mathematics (Calculus & Optimization)

The "from scratch" implementation relies heavily on calculus to minimize the error between the predicted values and the actual data points.

For a dataset with $m$ examples, the model attempts to fit a line described by the parameters $\theta$. The cost function, Mean Squared Error (MSE), is defined as:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

To find the optimal weights that minimize this cost function, we use gradient descent. We compute the partial derivative of the cost function with respect to each parameter $\theta_j$ using the chain rule:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

The parameters are then iteratively updated using a learning rate $\alpha$:

$$\theta_j \leftarrow \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

## 🚀 Getting Started

### Prerequisites

To run this notebook, you will need Python installed along with the following libraries:

* Python 3.x
* NumPy
* Scikit-Learn
* Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/Srihari0804/linear_regression.git
cd linear-regression

```


2. Install the required dependencies:
```bash
pip install numpy scikit-learn jupyter

```



## 💻 Usage

To explore the code and see the comparison in action, launch the Jupyter Notebook:

```bash
jupyter notebook calculus_w2.ipynb

```

Run the cells sequentially to observe the data setup, the scikit-learn model training, and the step-by-step mathematical breakdown of the from-scratch implementation.

## 📂 Project Structure

* `calculus_w2.ipynb` - The standalone notebook containing both the Scikit-Learn and NumPy implementations, along with data generation.

## 🎓 Acknowledgments

* This project was completed as part of the **Mathematics for Machine Learning: Calculus** course by DeepLearning.AI.
