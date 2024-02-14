<h1>Deep Learning Cryptocurrency Trading Model Trainer🤖🚀</h1>

<h2>
  Project Description:
</h2>
The DeepLearningCryptoTrading repository constitutes an encompassing model trainer designed to facilitate experimentation with diverse models and trainer classes. By default, the script generates a sliding window dataset with a sequence length of 60 minutes, leveraging the Alpaca-py library. This window incorporates a range of technical indicators, including but not limited to the Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD), each with distinct period lengths. The response variable hinges on the 8-minute SMA, wherein an instance is designated as 1 if the subsequent price surpasses the current instance price, and 0 if it falls below.<br><br>

Throughout the training process, a learning rate schedule is enforced with a threshold set at 10. The epoch yielding the highest F1-Score is preserved in the pretrained_models directory during pretraining and in the full_models directory during fine-tuning. A prospective expansion of this project entails formatting live data to construct the sliding window and establishing connections with various cryptocurrency exchanges for paper/live trading purposes. 


<h2>🏗️How to Use:</h2>

To install and run this project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HannibalLP33/DeepLearningCryptoTrading.git
   cd DeepLearningCryptoTrading
   ```
2. **Create Conda Environment**
      <br>Inside the Repository is an environment.yml file.  The virtual environment can be created using conda by following the steps below:
    ```bash
      conda create -name CustomEnv -f environment.yml
      conda activate CustomEnv
    ```
4. **Run main.py**
    <br>Using the provided config yaml file. Run the following:
     ```bash
       python main.py --cfg Model1_exp1.yaml
     ```
<h2>📖Manual</h2>

<h3>Folders📁</h3>
<ul>
  <li><strong>configs:</strong> Contains configuration files with model initialization parameters and training settings.</li>
  <li><strong>pretrained_models:</strong> Stores full model architectures and weights trained on a diverse range of currencies.</li>
  <li><strong>full_models:</strong> Holds full model architectures and weights fine-tuned on specific currencies.</li>
</ul>

<h3>Python Files🐍</h3>
<ul>
  <li><strong>common_funcs.py:</strong> Includes reusable functions utilized across the project.</li>
  <li><strong>main.py:</strong> Primary file orchestrating functionality by integrating <code>common_funcs</code>, <code>trainer</code>, and <code>model</code> modules. The primary training process occurs here.</li>
  <li><strong>trainer.py:</strong> Defines a custom trainer class responsible for setting up training and validation splits, initializing weights, defining metrics, and managing all aspects of the training process.</li>
  <li><strong>model.py:</strong> Contains the <code>DLModels</code> class housing various models for experimentation.</li>
</ul>


<h2>Ways to Contribute</h2>
1. <b>Code Contributions:</b> Help us improve the project by fixing bugs, implementing new features, or optimizing existing code.<br>
2. <b>Documentation:</b> Improve our README file, add code comments, or contribute to our project's documentation.<br>
3. <b>Testing:</b> Write and execute tests to ensure the reliability and stability of our codebase.<br>
4. <b>Feedback:</b> Share your thoughts, ideas, or suggestions for improvement by opening an issue or joining our discussions.<br>
5. <b>Spread the Word:</b> Help us reach a wider audience by sharing the DeepLearningTrading repository with others who might be interested.<br><br>

We value and appreciate your contributions, no matter how big or small. Thank you!

<h2>🚨Disclaimer🚨</h2>

<b>Risk Warning:</b> Cryptocurrency trading involves substantial risk of loss and is not suitable for everyone. Make sure you understand the risks involved and do thorough research before engaging in any trading activities.<br><br>
<b>Not Financial Advice:</b> This bot and its documentation do not constitute financial advice. The developers and contributors of this bot are not responsible for any losses incurred through the use of this repository.
