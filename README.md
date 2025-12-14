# Machine Learning Portfolio Projects

This portfolio showcases **machine learning algorithms implemented from scratch**. Each project includes complete implementations of algorithms from first principles, along with data preprocessing, model training, evaluation, and comprehensive analysis. All core algorithms are built from scratch without relying on high-level ML frameworks for the core implementations.

---

## Projects

### 1. Support Vector Machines (SVM)

**Description:**

**From-scratch implementation** of Support Vector Machines using the Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm. Complete `PegassosSVM` and `PegasosMinibatchSVM` classes built from first principles, including stochastic gradient descent, hinge loss subgradient computation, and weight projection.

**Key Features:**
- **Pegasos Algorithm**: Stochastic subgradient descent implementation from scratch
- **Hinge Loss**: Subgradient computation: $\partial_w L = \begin{cases} -\lambda w & \text{if } y_i(w^T x_i) \geq 1 \\ -\lambda w + y_i x_i & \text{otherwise} \end{cases}$
- **Weight Projection**: Projects weights onto ball: $||w|| \leq 1/\sqrt{\lambda}$ after each update
- **Custom Padder**: Feature padding with ones for bias term integration
- **Mini-Batch Variant**: Extension to mini-batch updates for reduced variance
- **Decision Function**: Signed distance computation: $f(x) = w^T x$ for classification

**Technical Highlights:**
- Stochastic gradient descent with learning rate: $\eta_t = \frac{1}{\lambda t}$
- Weight update: $w_{t+1} = (1 - \eta_t \lambda)w_t + \eta_t y_i x_i$ for misclassified samples
- Projection step: $w = \frac{w}{||w|| \sqrt{\lambda}}$ if $||w|| > 1/\sqrt{\lambda}$
- Label transformation from {0, 1} to {-1, 1} format
- Early stopping based on weight change tolerance

**Solution Notebook:** [SVM_Project.ipynb](./SVM_Project.ipynb)

---

### 2. Decision Trees

**Description:**

**From-scratch implementation** of decision tree regressor using variance reduction as the splitting criterion. Complete `TreeEstimator` class built from first principles with recursive tree construction, continuous feature handling, and greedy splitting algorithm.

**Key Features:**
- **Custom `TreeEstimator` Class**: Complete implementation from scratch with recursive DFS tree construction
- **Variance Reduction**: Splitting criterion: $\text{VarReduction} = \text{Var}(parent) - \left(\frac{n_{left}}{n} \cdot \text{Var}(left) + \frac{n_{right}}{n} \cdot \text{Var}(right)\right)$
- **Leaf Prediction**: Mean of target values: $\hat{y} = \frac{1}{|L|} \sum_{i \in L} y_i$ where $L$ is the set of samples in the leaf
- **Continuous Features**: Threshold-based splits for continuous features using unique values
- **Edge Case Handling**: Skips features with no variation, handles empty splits
- **Max Depth Control**: Prevents infinite recursion and overfitting

**Technical Highlights:**
- Recursive tree construction with depth-first search (DFS)
- Parent variance: $\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$
- Weighted child variance calculation for split evaluation
- Greedy algorithm that makes locally optimal splits
- Tree structure with `TreeNode` class containing decision functions and leaf labels

**Dataset:** Bike Sharing Dataset from UCI Machine Learning Repository for predicting daily bike rental counts

**Solution Notebook:** [DecisionTrees_Project.ipynb](./DecisionTrees_Project.ipynb)

---

### 3. Ensemble Learning

**Description:**

**From-scratch implementation** of ensemble methods: Random Forest (bagging) and AdaBoost (boosting). Complete `TreeEnsemble` and `AdaBoost` classes built from first principles, implementing bootstrap sampling, weighted error calculation, and ensemble prediction.

**Key Features:**
- **Random Forest (`TreeEnsemble`)**: Custom bagging implementation with bootstrap sampling and feature subsampling
- **Bootstrap Sampling**: Uses `np.random.RandomState` for reproducible sampling with replacement
- **Out-of-Bag (OOB) Error**: Efficient error estimation: $\text{OOB-MSE} = \frac{1}{n_{OOB}}\sum_{i \in OOB}(y_i - \hat{y}_i)^2$
- **AdaBoost**: Weighted bootstrap sampling with confidence calculation: $\alpha_t = \frac{1}{2}\log(\frac{1-\epsilon_t}{\epsilon_t})$
- **Prediction Flipping**: Handles weak learners with error > 0.5 by flipping predictions
- **Weight Updates**: Sample weight update: $w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))$ normalized to sum to 1

**Technical Highlights:**
- Bootstrap aggregation: each tree trained on random sample with replacement
- Ensemble prediction: $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$ for Random Forest
- Weighted voting: $\hat{y} = \text{sign}(\sum_{t=1}^{T} \alpha_t h_t(x))$ for AdaBoost
- OOB error handles empty rest_indexes gracefully
- Random state properly used for reproducibility

**Datasets:** 
- Bike Sharing Dataset for Random Forest regression
- Synthetic circles dataset for AdaBoost classification visualization

**Solution Notebook:** [EnsembleLearning_Project.ipynb](./EnsembleLearning_Project.ipynb)

---

### 4. Gradient Boosting

**Description:**

**From-scratch implementation** of gradient boosting for regression, focusing on L2Boost (gradient boosting with squared loss). Complete `L2GB` and `L2GBearly` classes built from first principles, implementing gradient descent in function space, residual fitting, and early stopping.

**Key Features:**
- **Custom `L2GB` Class**: Complete implementation from scratch with `fit()` and `predict()` methods
- **Initial Prediction**: Starts with mean of target values: $\hat{y}_0 = \frac{1}{n}\sum_{i=1}^{n} y_i$ for faster convergence
- **Residual Fitting**: Training weak learners on residuals: $r_i = y_i - F_{m-1}(x_i)$ where $F_{m-1}$ is the current ensemble
- **Ensemble Update**: Sequential model building: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$ where $\eta$ is learning rate
- **Early Stopping (`L2GBearly`)**: Automatic stopping based on validation performance with patience parameter
- **Input Validation**: Checks for compatible shapes, empty arrays, and valid hyperparameters

**Technical Highlights:**
- Gradient descent in function space: negative gradient of squared loss equals residuals
- Loss function: $L = \frac{1}{2}(y - F(x))^2$ with gradient: $-\nabla_F L = y - F(x)$
- Residual computation: $r_i = y_i - F_{m-1}(x_i)$ where each weak learner fits residuals
- Learning rate (shrinkage) $\eta$ prevents overfitting by controlling step size
- Early stopping monitors validation MSE with tolerance and patience

**Mathematical Foundation:**
- L2Boost minimizes squared loss through iterative residual fitting
- The negative gradient of squared loss equals the residual, making L2Boost elegant
- Initial prediction uses mean: faster convergence than starting from zero
- Ensemble prediction: $F_M(x) = \hat{y}_0 + \eta \sum_{m=1}^{M} h_m(x)$

**Dataset:** Bike Sharing Dataset from UCI Machine Learning Repository for predicting daily bike rental counts

**Solution Notebook:** [GradientBoosting_Project.ipynb](./GradientBoosting_Project.ipynb)

---

### 5. K-Nearest Neighbors (KNN)

**Description:** 

**From-scratch implementation** of the K-Nearest Neighbors algorithm for classification. Complete `KNNClassifier` class built from first principles using NumPy, implementing Euclidean distance computation, neighbor search, and majority voting without relying on ML libraries.

**Key Features:**
- **Custom `KNNClassifier` Class**: Complete implementation from scratch with `fit()`, `predict()`, and `score()` methods
- **Distance Computation**: Euclidean distance calculation using NumPy broadcasting: $d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
- **Neighbor Search**: Efficient k-nearest neighbor finding using `argpartition` for partial sorting
- **Input Validation**: Parameter validation ensuring k is positive and less than training set size
- **Decision Boundary Visualization**: Analysis of how k values affect model complexity and decision boundaries
- **Bias-Variance Tradeoff**: Comprehensive analysis demonstrating overfitting (small k) and underfitting (large k)

**Technical Highlights:**
- Lazy learning: stores entire training set, computes distances during prediction
- Efficient distance matrix computation using NumPy vectorization
- Majority voting with tie-breaking for classification
- Compatible with scikit-learn's `GridSearchCV` through proper estimator interface

**Dataset:** Synthetic voter party registration dataset with wealth and religiousness features

**Solution Notebook:** [KNN_Project.ipynb](./KNN_Project.ipynb)

---

### 6. Linear and Logistic Regression

**Description:**

**From-scratch implementation** of linear regression covering both analytical (closed-form) and iterative gradient descent solutions. Complete `OrdinaryLinearRegression` and `OrdinaryLinearRegressionGradientDescent` classes built from first principles, including custom `StandardScaler` and Ridge regression implementations.

**Key Features:**
- **Ordinary Least Squares (OLS)**: Analytical solution using matrix pseudo-inverse: $w = (X^T X)^{-1} X^T y$ with numerical stability considerations
- **Gradient Descent**: Custom iterative optimization with learning rate tuning and convergence monitoring
- **Ridge Regression**: L2 regularization implementation with analytical solution: $w_{Ridge} = (X^T X + \lambda I)^{-1} X^T y$
- **Custom StandardScaler**: Feature normalization implementation from scratch for gradient descent stability
- **Convergence Detection**: Automatic stopping based on weight change tolerance and divergence detection
- **Regularization Comparison**: Analysis of Ridge (L2) vs Lasso (L1) regularization effects on coefficients

**Technical Highlights:**
- Matrix operations using `np.linalg.pinv()` for numerical stability with rank-deficient matrices
- Gradient computation: $\nabla_w L = \frac{2}{N}X^T(Xw - y)$ for squared loss
- Convergence check: stops when $||w_{t+1} - w_t|| < \text{tol}$
- Divergence detection: warns when loss increases significantly (10x)
- Proper train/test split handling to prevent data leakage in normalization

**Dataset:** Diabetes regression dataset from scikit-learn, predicting disease progression based on physiological measurements

**Solution Notebook:** [Regression_Project.ipynb](./Regression_Project.ipynb)

---

### 7. Clustering

**Description:**

**From-scratch implementations** of three fundamental unsupervised learning algorithms: **K-Means** (partitional clustering), **Gaussian Mixture Models** (probabilistic clustering), and **DBSCAN** (density-based clustering). Custom `LloydsKMeans` class built from first principles, with comprehensive analysis of different clustering approaches.

**Key Features:**
- **K-Means (`LloydsKMeans`)**: Custom implementation of Lloyd's algorithm from scratch
- **Objective Function**: Minimizes $\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$ where $\mu_i$ is the centroid of cluster $C_i$
- **Centroid Update**: $\mu_i = \frac{1}{|C_i|}\sum_{x \in C_i} x$ after each assignment step
- **Gaussian Mixture Models**: Probabilistic clustering using Expectation-Maximization (EM) algorithm
- **DBSCAN**: Density-based clustering identifying core points, border points, and noise
- **Kernel Methods**: Polynomial kernel transformation for non-linearly separable data (XOR pattern)

**Technical Highlights:**
- Lloyd's algorithm: iterative assignment and centroid update until convergence
- EM algorithm: E-step computes responsibilities, M-step updates parameters
- DBSCAN: density reachability and core point identification
- Kernel transformation: $K(x, y) = (x^T y + c)^d$ for polynomial features
- Cluster evaluation using elbow method and silhouette analysis

**Dataset:** Electronic Medical Records (EMR) containing blood test results (Na, K, ALT, AST, WBC, RBC, Hgb, Hct) from patients, plus synthetic datasets for algorithm comparison

**Solution Notebook:** [Clustering_Project.ipynb](./Clustering_Project.ipynb)

---

### 8. Naive Bayes

**Description:**

**From-scratch implementation** of Multinomial Naive Bayes classifier for text classification. Complete `NaiveBayes` class built from first principles using Bayes' theorem, implementing log-probability calculations, Laplace smoothing, and sparse matrix handling.

**Key Features:**
- **Custom `NaiveBayes` Class**: Complete implementation from scratch with `fit()`, `predict()`, and `predict_log_proba()` methods
- **Bayes' Theorem**: Posterior probability calculation: $P(c|d) = \frac{P(c) \cdot P(d|c)}{P(d)}$ with naive independence assumption
- **Laplace Smoothing**: Zero probability handling: $P(w_i|c) = \frac{\text{count}(w_i, c) + \alpha}{\sum_{j} \text{count}(w_j, c) + \alpha \cdot |V|}$
- **Log-Space Computation**: Prevents numerical underflow: $\log P(c|d) = \log P(c) + \sum_{i=1}^{n} \log P(w_i|c)$
- **Sparse Matrix Support**: Handles both dense and sparse matrices using `scipy.sparse.issparse`
- **Custom Preprocessor**: Text preprocessing with lemmatization and vectorization integration

**Technical Highlights:**
- Prior probability: $P(c) = \frac{N_c}{N}$ where $N_c$ is number of documents in class $c$
- Likelihood computation with sparse matrix operations for memory efficiency
- Prediction: $\hat{c} = \arg\max_{c} \log P(c|d)$ using log-probabilities
- Feature count computation: $\text{count}(w_i, c) = \sum_{d \in c} \text{count}(w_i, d)$

**Dataset:** 20 newsgroups dataset containing approximately 18,000 newsgroup posts covering 20 topics with realistic temporal train/test split

**Solution Notebook:** [NaiveBayes_Project.ipynb](./NaiveBayes_Project.ipynb)

---

### 9. Reinforcement Learning

**Description:**

**From-scratch implementation** of reinforcement learning algorithms focusing on Multi-Armed Bandits (MABs). The project implements exploration-exploitation strategies, demonstrating the fundamental tradeoff between trying new actions and exploiting known good actions.

**Key Features:**
- **Multi-Armed Bandits**: Implementation of the classic exploration-exploitation problem
- **Epsilon-Greedy Algorithm**: Simple strategy balancing exploration and exploitation
- **Upper Confidence Bound (UCB)**: More sophisticated strategy using confidence intervals
- **Regret Analysis**: Measuring performance through cumulative regret over time
- **Strategy Comparison**: Comparing different exploration-exploitation approaches
- **Visualization**: Plotting reward accumulation and regret over time

**Technical Highlights:**
- Exploration vs exploitation tradeoff analysis
- Confidence interval calculations for UCB algorithm
- Regret minimization as performance metric
- Stochastic and adversarial bandit scenarios
- Real-world applications in recommendation systems and A/B testing

**Solution Notebook:** [ReinforcementLearning_Project.ipynb](./ReinforcementLearning_Project.ipynb)

---

## Technical Standards

- **All core algorithms are implemented from scratch** using NumPy and core Python libraries
- Each project notebook contains complete implementations with proper documentation
- Projects include data preprocessing, model training, evaluation, and visualization
- Code follows PEP 8 standards with type hints and docstrings (Google/NumPy style)
- Inline comments for complex ML logic, model architecture decisions, and hyperparameter choices
- Clear separation of concerns: data loading, preprocessing, model definition, training, evaluation
- All projects are ready for portfolio presentation
- Implementations demonstrate deep understanding of algorithm mechanics and mathematical foundations
