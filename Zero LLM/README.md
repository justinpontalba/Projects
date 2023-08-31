# Test task Zero LLM
In the financial sector, one of the routine tasks is mapping data from various sources in Excel tables. For example, a company may have a target format for employee health insurance tables (Template) and may receive tables from other departments with essentially the same information, but with different column names, different value formats, and duplicate or irrelevant columns.
Your task is to devise an approach using LLM for mapping tables A and B to the template by transferring values and transforming values into the target format of the Template table (example below)
 
Example tables are located in the /data folder as .csv files.

## Possible solution approach:
1.	Extract information about the columns of the Template table and tables A and B in the format of a text description.
2.	For each of the candidate tables (A and B), ask the LLM to find similar columns in the Template.
3.	In case of ambiguous mapping, ask the user to choose the most suitable column from the candidates.
4.	Automatically generate data mapping code for each column display in the final Template format. For example, for date columns, they may be in different formats, and it is necessary to change the order from dd.mm.yyyy to mm.dd.yyyy. The person can check the code (or pseudocode) that will perform the mapping.
5.	Check that all data has been transferred correctly; if any errors occur, issue an alert to the person.
## There is also an additional challenge* (Not required but would be a big plus):
Since such operations can be repeated quite often, and a person will edit the transformation logic, it is desirable to save this data and have the ability to retrain on it. Propose an approach for retraining and try to implement such retraining on synthetic examples (you can come up with them using GPT)
## Expected result:
1. Code on GitHub using OpenAI API, Langchain, and other LLMOps tools of your choice.
2. An interface in which you can perform such an operation. Place the interface in the public domain for testing work
### Example user journey:
- The user uploads the Template table.
- The user uploads table A.
- For each column in the Template, the system suggests columns from column A (1 or more relevant candidates), showing the basis for the decision (formats, distributions, and other features that are highlighted in the backend).
- The person confirms the mapping.
- Next, the data transformation stage begins. The system generates and displays the code with which it will perform the transformation. The user can edit it and run it, checking the correctness of the mapping.
- At the output, the user receives a table in the Template format (columns, value formats) but with values from table A.
- The same for table B.
3. Your thoughts on edge cases and how they can be overcome

# Implementation
### Requirememts
- python==3.11.4

### Set-up
```
conda env create -f environment.yml
conda activate zerollm
```
```
pip install -r requirements.txt
```
### Run app
```
streamlit run interface.py     
```
## Results

Successfully Uploading Table A and B with generate descriptions

**Table A**
![image](https://github.com/justinpontalba/Projects/assets/58340716/33b2488d-8cbf-4ced-9772-08b9b3a62d9a)

**Table B**

![image](https://github.com/justinpontalba/Projects/assets/58340716/cf8b6252-6a61-4a9c-8dba-1cd3e943cfb3)

### Column Candidate Suggestions
![image](https://github.com/justinpontalba/Projects/assets/58340716/68e2f5a0-6aa0-4052-ab77-87e92d46c149)

### Transformation mapping generation
![image](https://github.com/justinpontalba/Projects/assets/58340716/0d1b8cd4-abea-4248-b9d4-439fb16934dd)

### Applying Transformations
![image](https://github.com/justinpontalba/Projects/assets/58340716/623428f0-4800-497b-accc-e882709249ad)

## Edge Cases
Tabular data across departments and organizations can be highly unstructured. Some edge cases are:
- **Handling missing data:** Empty Tables and Missing Columns are few examples that can be encountered. Empty rows when read into a python environment can be interpretted as NaNs. Dropping empty rows, or filling in empty values with a place holder such as "[Empty]" is a way to recognize that data is missing without have to drop data points.
- **Special characters:** Tabular data can be filled with characters encoded/decoded incorrectly resulting in incomprehensible data. Ensuring standardized encoding methods for reading or preprocessing can be considered.
- **Data Volume/Token Limitation:** If generative models are the intended approach for performing transformation tasks, LLMs especially in the open-source (LLama2, Falcon) have lower limits than paid services like OpenAI. Strategically chunking data can be used to work within the limits of LLMs.
- **LLM Unpredictability despite prompt engineering:** While not extensive, throughout the exercise various prompts were used to ensure that transformations were being identifed and applied accordingly. However, GPT hallucinated at times or had inconsistent outputs, making post processing a lot more difficult. Error handling hallucinations through semantic comparison may be an approach to catch when LLMs hallucinate. 
- **Scaling generative models:** If generative models are the direction, scaling an architecture to handle larger volumnes of data + requests (if hosting yourselves) is a consideration. Containerizing an API end point with an open source model can be scaled both vertically and laterally.
- **From a security stand point, code injection:** If LLMs are being used locally, there is a potential for code injection if the user has malicious intent when having the ability to edit transformation instructions. Using LLMs like Llama2 which have a "safety" component built into it can mitigate risks related to code injection. Other strategies include input validation, Parameterized Queries, Security Libraries etc.
- **Language Translation:** Textual data may contain other languages. Most LLMs perform better on enlish data, or exclusively on english data. Supporting the Table reading process by leveraging translation in preprocessing can increase the coverage of this exercises' goal.

## Additional Challenge
**Proposed Approach:**
1. **Generate Synthetic Data:** Create synthetic data that mimics the structure and statistical characteristics of real financial data. 
2. **Mask Sensitive Information:** Replace data points in your real financial data with masked values or placeholders. This could involve replacing actual names, account numbers, or other sensitive details with fictional placeholders.
3. **Model Development:** Develop your data transformation logic and machine learning models using the masked data. Since the masked data retains the structure of the original data, your models can still learn the relevant patterns and relationships.
