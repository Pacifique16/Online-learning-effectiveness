# **📊 Academic Performance in Online Learning**

<br>

 **NAME**:   Pacifique HARERIMANA
 <br>
 **ID**:        26937
 <br>
 **CONCENTRATION**:   Software Engineering

---
<br>

## 📁 Dataset Source

This project uses the **"Academic Performance in Online Learning"** dataset from [Figshare](https://figshare.com/articles/dataset/Academic_Performance_in_Online_Learning/28238927). It provides detailed CSV files capturing students’ demographics, registration, assessments, and VLE interaction.

**CSV Files Used:**

* `assessments.csv`
* `courses.csv`
* `studentInfo.csv`
* `studentRegistration.csv`
* `vle.csv`
* `studentVle.csv`
* `studentAssessment.csv`

<br>

# **PHASE 1: 🧠 Problem Statement**

### 🎯 Problem

Many online learners underperform due to a lack of motivation, poor engagement, or untracked study habits. Institutions struggle to identify early predictors of failure in e-learning settings.

### ✅ Goal

To predict student performance and analyze how engagement (click activity) and demographics influence academic success.

<br>
<br>

# **PHASE 2: 📥 Data Loading and Preview**

### 🔸 Code to Load CSV Files into Pandas

```python
import pandas as pd

# Load datasets
folder_path = r"C:\Users\Pacifique Harerimana\Downloads\sabiya\Dataset\\"

assessments = pd.read_csv(folder_path + "assessments.csv")
courses = pd.read_csv(folder_path + "courses.csv")
studentInfo = pd.read_csv(folder_path + "studentInfo.csv")
studentRegistration = pd.read_csv(folder_path + "studentRegistration.csv")
vle = pd.read_csv(folder_path + "vle.csv")
studentVle = pd.read_csv(folder_path + "studentVle.csv")
studentAssessment = pd.read_csv(folder_path + "studentAssessment.csv")
```
<br>

### 🔸 Code to Preview each file

```python
print("studentInfo.csv:")
print(studentInfo.head())

print("\nassessments.csv:")
print(assessments.head())
```
<br>

Result: 

<img width="1212" height="661" alt="Preview Each File" src="https://github.com/user-attachments/assets/e34c8b6b-2982-4e51-82b2-cf9062320833" />

<br>
 <BR>
 
 ###### _Explanation:_
 Each file is loaded into a separate pandas DataFrame for inspection and cleaning.
<br>

---

<br>

# **PHASE 3: 🔍 Data Cleaning - Missing Values & Data Types Check**

### 🔸 Code – Missing Values

```python
dataframes = {
    'assessments': assessments,
    'courses': courses,
    'studentInfo': studentInfo,
    'studentRegistration': studentRegistration,
    'vle': vle,
    'studentVle': studentVle,
    'studentAssessment': studentAssessment
}

for name, df in dataframes.items():
    print(f"\n{name} missing values:\n{df.isnull().sum()}")

```

<br>

Result: 

<img width="1072" height="832" alt="Check for Missing Values" src="https://github.com/user-attachments/assets/84701eac-be27-4a47-a4a2-48265266cc73" />

<br>

### 🔸 Code – Data Types

```python
# Checking data types
for name, df in dataframes.items():
    print(f"\n{name} data types:\n{df.dtypes}")

```

<br>

Result: 
<img width="1233" height="840" alt="Check Data Types   Consistency" src="https://github.com/user-attachments/assets/90016f9c-9697-4e1d-be24-260bae317d62" />

 <BR>
 
 ###### _Explanation:_
 * Found missing values in `imd_band`, `date_unregistration`, `score`, etc.
 * Converted necessary fields from float to integer for date consistency.
<br>

---

<br>

# **PHASE 4: 🛠 Data Preprocessing**

### 🔸 Code handling missing data

```python
# Clean assessments
assessments.dropna(subset=['date'], inplace=True)
assessments['date'] = assessments['date'].astype(int)
assessments['weight'] = assessments['weight'].astype(int)

# Clean studentInfo
studentInfo['imd_band'].fillna('Unknown', inplace=True)

# Clean studentRegistration
studentRegistration.dropna(subset=['date_registration'], inplace=True)
studentRegistration['date_registration'] = studentRegistration['date_registration'].astype(int)
studentRegistration['date_unregistration'] = studentRegistration['date_unregistration'].fillna(-1).astype(int)

# Clean vle
vle[['week_from', 'week_to']] = vle[['week_from', 'week_to']].fillna(-1).astype(int)

# Clean studentAssessment
studentAssessment['score'] = studentAssessment['score'].fillna(0)
```
<br>

 ###### _Explanation:_

Missing data handled to avoid model errors. For example, -1 was used to indicate no unregistration or undefined time frames.

---

<br>
<br>

# **PHASE 5: 📊 Exploratory Data Analysis (EDA)**

### 🔸 Code – Gender Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=studentInfo, x='gender', hue='final_result')
plt.title("Gender vs Final Result")
plt.show()
```
<br>

Result: 
<img width="916" height="582" alt="Visualize Relationships (using matplotlib or seaborn), final result by gender" src="https://github.com/user-attachments/assets/c474a75a-0a7e-4dcc-8332-b01ca5a97b13" />

 <BR>
 
 ###### _Explanation:_
Shows how male and female students performed. Helps spot performance trends by gender.
<br>

---

### 🔸 Code – Age Band vs Final Result

```python
sns.countplot(data=studentInfo, x='age_band', hue='final_result')
plt.title("Age Band vs Final Result")
plt.show()
```
<br>

Result: 
<img width="917" height="582" alt="Age Band vs Final Result" src="https://github.com/user-attachments/assets/ee8afad0-8e53-47dc-b667-ad91a7ab7a38" />

 <BR>
 
 ###### _Explanation:_
Most students who passed fall into the 26–35 age band.
<br>

---

### 🔸 Code – Distribution of Assessment Scores

```python
sns.histplot(data=studentAssessment, x='score', bins=20, kde=True)
plt.title('Distribution of Assessment Scores')
plt.show()
```
<br>
Result: 

<img width="937" height="573" alt="Score Distribution in Assessments" src="https://github.com/user-attachments/assets/b07115f0-1650-404a-8394-62d44434f210" />

 <BR>

---

<br>

# **PHASE 6: 🔗 Dataset Merging & Feature Engineering**

### 🔸 Code

```python
# Merge studentInfo with studentAssessment
student_scores = pd.merge(studentInfo, studentAssessment, on='id_student', how='left')

# Add total clicks per student
clicks_per_student = studentVle.groupby('id_student')['sum_click'].sum().reset_index()
clicks_per_student.rename(columns={'sum_click': 'total_clicks'}, inplace=True)

# Final merge
student_scores_clicks = pd.merge(student_scores, clicks_per_student, on='id_student', how='left')

# Fill missing clicks with 0
student_scores_clicks['total_clicks'] = student_scores_clicks['total_clicks'].fillna(0)

# Save to CSV
student_scores_clicks.to_csv("final_dataset.csv", index=False)
```
<br>

Result: 

<img width="877" height="83" alt="final dataset" src="https://github.com/user-attachments/assets/90dc97bb-46b3-4e29-a96f-3b0d36fa38d0" />

 <BR>
 
 ###### _Explanation:_
Created a single, clean dataset with assessment scores, student demographics, and VLE interaction data.
<br>

---

<br>

# **PHASE 7: 🤖 Modeling – Predicting Student Performance**

### 🔸 Code – Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


X = data.drop('final_result', axis=1)
y = data['final_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

Result: 
<img width="697" height="263" alt="accuracy" src="https://github.com/user-attachments/assets/87e0f38b-6e8d-49ff-a065-3f9dc9ec373d" />


 <BR>
 
 ###### _Explanation:_
Used Random Forest Classifier to predict final results. Model showed decent accuracy on test data and identified important features.

<br>

---

<br>

# **PHASE 8: 📊 Power BI Dashboard**

### 🖼️ Screenshot 1 – Gender vs Result

<img width="1166" height="711" alt="Gender vs Result" src="https://github.com/user-attachments/assets/85baa866-417d-4033-b397-c39641284454" />


### 🖼️ Screenshot 3 – Age Band and Clicks

<img width="1167" height="705" alt="Age Band and Clicks" src="https://github.com/user-attachments/assets/e373cd89-7b48-4e59-ad55-b05b4ea21435" />

###### _Explanation:_
Power BI visualized the performance by demographics and click interaction, making trends easier to communicate and explore.
* For More Visuals, i have shared the Power BI file 
<br>


---

# ✅ Conclusion

* Students with higher engagement (more clicks) tend to pass.
* Younger students (26–35) performed best overall.
* Random Forest classified performance with reasonable accuracy.

---
