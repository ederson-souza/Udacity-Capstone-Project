# Machine Learning Engineer Nanodegree
## Capstone Proposal
## TalkingData AdTracking Fraud Detection Challenge
Nguyen Quoc Bao  
June 17, 2018

## Proposal
<!-- _(approx. 2-3 pages)_ -->

### Domain Background
<!-- _(approx. 1-2 paragraphs)_ -->

<!-- In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required. -->

Presently, fraud traffic is one of the most pressing matters coupled with the exponentially growing mobile phone prevalence. With the number of smart mobile devices approaching 2 billion worldwide, frauds in the mobile app ecosystem are more and more rampant.

<div style="align: center; text-align: center;">
![fraud rate 2017](assets/fraud_rate_graph.png)
<p>
_Source: [DataVisor](https://www.datavisor.com/)_
</p>
</div>

It is shown that fraudulent apps led to at least 15% of the total amount of apps downloaded and installed on smartphones in 2017. In 2016, $5.7 billion dollars were spent on mobile advertising in the US alone; these numbers tell us that fraudsters are making away with hundreds of millions from developers and businesses each year. Clearly, mobile frauds are a severe financial burden and a serious threat to the mobile industry.  

### Problem Statement
<!-- _(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once). -->

[Talking Data](https://www.talkingdata.com/) is one of the most prominent big data platform, covering more than 70% of mobile devices' activities across China. They handle billions of ads clicks on mobile per day that are potentially fraudulent. Their current strategy against mass-install factories and fake click producers is to build a portfolio for each IP address and device based on the behavior of the click, flagging those who produce lots of clicks but never end up installing apps.

In their [Kaggle competition](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection), Talking Data provided a dataset detailing the clicks registered by their system; the goal is to build a model that predicts whether an user will download an app after clicking a mobile app or not. This model would help the firm increase their solution's accuracy in identifying fraudsters.

### Datasets and Inputs
<!-- _(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem. -->

The dataset provided by Talking Data on [Kaggle competition homepage](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) includes approximately 200 million registered clicks over 4 days, split into training and testing sets. The training set contains more than 180 million rows of data, each has the timestamp of the click, number-encoded IP addresses, device numbered label code, device's operating system code, app code, channel code, whether the click resulted in a download or not, and time of download if applicable. The testing set contains about 18 million clicks with each click associated with an ID and other information excluding the download or not label and download time.

### Solution Statement
<!-- _(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once). -->

The solution is a classification model capable of predicting whether a click with known attributes would result in the respective app being downloaded to the device or not. First, I will use Pandas and Numpy to gain some understanding of the data, then try to devise some new features based on the given features to train on. As for model, I am inclined towards XGBoost, a powerful Gradient Boosting framework that has proven itself in many past Kaggle competitions while being versatile and work with other frameworks such as scikit-learn.

### Benchmark Model
<!-- _(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail. -->

Randomly predicting with equal probability that whether or not a click results in a download with absolute certainty yields a ROC-AUC score of 0.3893 on the test dataset.

### Evaluation Metrics
<!-- _(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms). -->

A model in this competition is graded based on the area-under-the-ROC-curve score between the predicted class probability and the observed target, measured on the test data. Since the test data is not labelled, grading is done by uploading the file containing the download probability of each click in the test data to Kaggle.

### Project Design
<!-- _(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project. -->

* **Programming language:** Python 3.6
* **Library:** Pandas, Numpy, Scikit-learn, XGBoost
* **Workflow:**
  * Establish basic statistics and understanding of the dataset; perform basic cleaning and processing if needed.
  * Train a base classification model on the given data as-is to gauge the performance.
  * Devise new features based on the given features; some intuitions: what day of the week the click was registered, what time of day, is it a holiday or is there any social event going on, how often a specific user or IP address invokes a click, etc.
  * Fine tune the model's hyperparameters.
  * Perform training.

### Prospects

The sheer size of the given data (over 180 million rows of data, ~7.3GB) will no doubt pose a challenge to both the data processing and training steps. Having to upload the test prediction in order to assess performance at every round will also be a hurdle since the process will take a lot of time.

-----------

<!-- **Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced? -->
