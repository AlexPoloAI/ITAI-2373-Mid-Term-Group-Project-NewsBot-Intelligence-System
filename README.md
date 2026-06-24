# ITAI-2373-Mid-Term-Group-Project-NewsBot-Intelligence-System
I'm working on this project by myself, without a group.
https://eagleonline.hccs.edu/courses/332496/assignments/8524360

6263-14054-ITAI-2373AssignmentsMid-Term Group Project: NewsBot Intelligence System

Mid-Term Group Project: NewsBot Intelligence System 
Due Saturday by 11:59pm Points 100 Submitting a file upload
Project Overview
Welcome to your mid-term group project! This comprehensive assignment challenges you to build a NewsBot Intelligence System that demonstrates mastery of all NLP techniques covered in Modules 1-8. Working in teams of 3-4 students, you will create an end-to-end system that automatically processes, categorizes, and extracts insights from news articles.
This project requires you to actively integrate multiple NLP techniques into a cohesive, working system. The goal is to use coding as a tool to solve real-world problems and build your analytical and system design skills.

A student notebook  is available in hte module Midterm to help you start. 

Duration: This project spans one week and will likely require 15-20 hours per team to complete thoughtfully, with work distributed among team members.

⭐ Why This Project Matters
News intelligence systems power many applications you interact with daily. They enable Google News to categorize articles, help financial firms monitor market sentiment, allow media companies to track competitor coverage, and enable researchers to analyze public opinion trends. Building this system will give you hands-on experience with the complete NLP pipeline used in industry applications for media monitoring, business intelligence, and automated content management.

🎓 Learning Objectives
By completing this project, you will be able to:

Integrate and Apply all NLP techniques from Modules 1-8 into a unified, working system
Design and Implement a complete text processing pipeline from raw input to actionable insights
Compare and Evaluate different machine learning algorithms for text classification tasks
Extract and Analyze structured information (entities, sentiment, patterns) from unstructured text
Generate Business Value by translating technical NLP results into meaningful business insights
Collaborate Effectively on a complex software project using industry-standard tools and practices
Communicate Technical Results to both technical and business audiences through documentation and presentations
📖 Core System Components (Must Include All)
Module 1: Real-World NLP Application Context
Clear business case and application description, Industry context and use case analysis, Target user identification and value proposition
Module 2: Text Preprocessing Pipeline
Comprehensive text cleaning and normalization, Tokenization, stop word removal, and lemmatization, Handling of various text formats and edge cases
Module 3: TF-IDF Feature Extraction and Analysis
TF-IDF vectorization with parameter optimization, Category-specific term analysis and visualization, Feature importance analysis and interpretation
Module 4: Part-of-Speech Pattern Analysis
POS tagging and grammatical pattern extraction, Cross-category POS distribution analysis, Writing style differences identification
Module 5: Syntax Parsing and Semantic Analysis
Dependency parsing and relationship extraction, Syntactic feature engineering, Semantic role identification
Module 6: Sentiment and Emotion Analysis
Article-level sentiment classification, Emotional tone analysis and interpretation, Sentiment distribution across news categories
Module 7: Multi-Class Text Classification System
Implementation of multiple classification algorithms, Comprehensive model evaluation and comparison, Performance optimization and feature selection
Module 8: Named Entity Recognition and Analysis
Entity extraction and classification (PERSON, ORG, GPE, DATE, MONEY), Entity frequency analysis and relationship mapping, Cross-category entity pattern analysis
🎯 Project Goals
Integration: Combine all Module 1-8 techniques into one cohesive system
Real-World Application: Build something portfolio-worthy for your professional development
Computational Efficiency: Work within Google Colab free tier limitations
Team Collaboration: Develop professional software development skills
📊 What You'll Build
Your NewsBot Intelligence System will:

✅ Preprocess news articles using advanced text cleaning techniques
✅ Classify articles into categories (Politics, Sports, Technology, Business, Entertainment, Health)
✅ Extract entities (people, organizations, locations, dates, money amounts)
✅ Analyze sentiment and emotional tone of news content
✅ Identify patterns using TF-IDF, POS tagging, and syntax parsing
✅ Generate insights for media monitoring and business intelligence
💻 Technical Requirements
Platform: Google Colab Free Tier (no paid resources required)
Libraries: scikit-learn, spaCy (small models), NLTK, pandas, matplotlib
Dataset: Provided curated news dataset (1,000 articles) OR approved alternative
Code Quality: Professional-level documentation and organization
📊 Dataset Acquisition from Kaggle (Primary Method)
You will use Kaggle datasets for this project. Follow these complete, step-by-step instructions:

Step 1: Create Kaggle Account (If You Don't Have One)
Go to https://www.kaggle.com
Click "Register" in the top right corner
Create account using email or Google/Facebook login
Verify your email address
Complete your profile (optional but recommended)
Step 2: Choose Your Dataset
Select ONE of these recommended datasets:

BBC News Classification Dataset (RECOMMENDED)

URL: https://www.kaggle.com/competitions/learn-ai-bbc/data
Categories: Business, Entertainment, Politics, Sport, Tech
Size: ~2,000 articles
Perfect for: Beginners, clear categories, manageable size
News Category Dataset
URL: https://www.kaggle.com/datasets/rmisra/news-category-dataset
Categories: Multiple news categories from HuffPost
Size: Large (you'll need to sample to 1,000-2,000 articles)
Perfect for: More variety, real-world data
All the News Dataset

URL: https://www.kaggle.com/datasets/snapcrack/all-the-news
Categories: Various news sources
Size: Very large (requires significant sampling)
Perfect for: Advanced students, diverse sources
Step 3: Get Your Kaggle API Key
To download datasets programmatically in Google Colab, you need an API key:

Go to https://www.kaggle.com/account
Scroll down to the "API" section
Click "Create New API Token"
A file named "kaggle.json" will download to your computer
Remember where this file is saved - you'll need it in Google Colab
Step 4: Download Dataset in Google Colab
Method 1: Using Kaggle API (Recommended for Learning)

Copy and paste this code into your Google Colab notebook:

# Step 1: Install Kaggle API
!pip install kaggle

# Step 2: Upload your kaggle.json file
from google.colab import files
print("Please upload your kaggle.json file:")
uploaded = files.upload()

# Step 3: Set up API credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle API setup complete!")

For BBC News Dataset (recommended):

# Download BBC News Dataset
!kaggle competitions download -c learn-ai-bbc

# Unzip the files
!unzip learn-ai-bbc.zip

# List the files to see what we have
!ls -la

# Load the dataset
import pandas as pd
import os

# Check what files are available
print("Available files:")
for file in os.listdir('.'):
    if file.endswith('.csv'):
        print(f"  - {file}")

# Load the main dataset (adjust filename as needed)
# Common filenames: train.csv, bbc-text.csv, or similar
df = pd.read_csv('train.csv')  # Adjust filename based on what you see
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Categories: {df['category'].unique()}")  # Adjust column name as needed

For News Category Dataset:

# Download News Category Dataset
!kaggle datasets download -d rmisra/news-category-dataset

# Unzip the files
!unzip news-category-dataset.zip

# Load and sample the dataset (it's large!)
import pandas as pd
import json

# Load the JSON file
with open('News_Category_Dataset_v3.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Sample to manageable size for Colab
df_sample = df.sample(n=2000, random_state=42)

print(f"Original dataset shape: {df.shape}")
print(f"Sampled dataset shape: {df_sample.shape}")
print(f"Categories: {df_sample['category'].unique()}")

Method 2: Direct Download (Alternative)

Go to your chosen dataset URL
Click the blue "Download" button
Extract the ZIP file on your computer
In Google Colab, click the folder icon on the left sidebar
Click the upload button and upload your CSV file
Load the dataset using: df = pd.read_csv("your_filename.csv")
Step 5: Prepare Your Dataset
Once you have your dataset loaded, prepare it for the project:

# Data preparation template
import pandas as pd
import numpy as np

# 1. Examine your dataset
print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# 2. Identify text and category columns
# Adjust these column names based on your dataset
text_column = 'text'  # or 'description', 'content', 'headline', etc.
category_column = 'category'  # or 'label', 'class', etc.

# 3. Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# 4. Remove rows with missing text or categories
df_clean = df.dropna(subset=[text_column, category_column])

# 5. Check category distribution
print(f"\nCategory distribution:")
print(df_clean[category_column].value_counts())

# 6. Sample if dataset is too large (keep under 2000 for Colab)
if len(df_clean) > 2000:
    df_final = df_clean.sample(n=2000, random_state=42)
    print(f"\nSampled dataset to {len(df_final)} rows")
else:
    df_final = df_clean

# 7. Rename columns for consistency
df_final = df_final.rename(columns={
    text_column: 'content',
    category_column: 'category'
})

# 8. Save prepared dataset
df_final.to_csv('newsbot_dataset.csv', index=False)
print("\n✅ Dataset prepared and saved as 'newsbot_dataset.csv'")

Dataset Requirements (MUST MEET ALL)
Minimum 500 articles across at least 4 categories
Text content must be substantial (not just headlines)
Clear category labels for classification
English language articles only
Maximum 2,000 articles (to stay within Colab limits)
No missing values in text or category columns
Common Issues and Solutions
Error: "403 Forbidden"
Make sure you've accepted the dataset's terms and conditions on Kaggle website
Error: "Dataset too large for Colab"
Use the sampling code provided above to reduce dataset size
Error: "Kaggle API not working"
Use Method 2 (direct download) instead
Error: "Column not found"
Check your dataset columns and adjust column names in the code
Error: "Out of memory"
Reduce sample size to 1,000 articles or less
🚀 Getting Started Instructions
Step 1: Individual Repository (each of you should have your individual repository  up from previous assignments and for the  porftfolio)
Teams are already formed - work with your assigned group members
Each student uploadsthe group work to their own individual GitHub portfolio repository
Create a new folder in your portfolio repository: ITAI2373-NewsBot-Midterm
Download the provided starter notebook template and add it to your repository folder
Ensure your repository is public so the instructor can access and test your project
Step 2: Project Planning and Role Assignment
Meet with your team to review the complete project requirements
Assign primary responsibility for each module to team members
Establish regular check-in meetings and communication protocols
Plan how you will coordinate work while each maintaining individual repositories
Step 3: Development Environment Setup
Ensure all team members can access and run the starter notebook in Google Colab
Test the provided dataset loading and basic functionality
Verify all required libraries can be installed and imported successfully
Coordinate with team members on shared code components and integration
❗ Important Notes for Students
Focus on Integration and Analysis
This is a comprehensive systems project where the primary goal is to demonstrate how multiple NLP techniques work together to solve real problems. Coding is the vehicle for your understanding, but the analysis, integration, and insights are where you demonstrate mastery.

AI Code Assistance is Permitted
You are encouraged to use AI tools as a learning aid to help you understand syntax, debug issues, or explore implementation approaches. However, the system design, analysis, and reflection must be your own work. The assessment heavily weights your ability to integrate techniques, interpret results, and articulate insights.

Computational Resource Management
All work must be completed using free computational resources. Design your experiments and data processing to work within Google Colab's limitations. Use efficient algorithms, process data in batches if necessary, and save intermediate results to avoid recomputation.

Professional Development Focus
This project is designed to be portfolio-worthy. Approach it with the same professionalism you would bring to a workplace project. Your GitHub repository should showcase your technical skills to potential employers.

 

📊 Dataset Information and Acquisition
 Kaggle Datasets

To use  a Kaggle dataset, follow these EXACT steps:

Step 1: Find Suitable Datasets on Kaggle
Recommended Datasets:

BBC News Classification Dataset - URL: https://www.kaggle.com/competitions/learn-ai-bbc/data
News Category Dataset - URL: https://www.kaggle.com/datasets/rmisra/news-category-dataset
All the News Dataset - URL: https://www.kaggle.com/datasets/snapcrack/all-the-news
Step 2: Download Dataset from Kaggle
Method 1: Direct Download (Easiest)

Go to the dataset URL (e.g., BBC News Classification)
Click the blue "Download" button (you may need to create a free Kaggle account)
Extract the ZIP file to get CSV files
Upload the CSV file to your Google Colab session
Method 2: Using Kaggle API in Google Colab - See detailed code examples in the full instructions.

Step 3: Get Your Kaggle API Key
Go to https://www.kaggle.com/account
Scroll down to "API" section
Click "Create New API Token"
Download the kaggle.json file
Upload this file when prompted in Google Colab
Dataset Requirements (MUST MEET ALL)
Minimum 500 articles across at least 4 categories
Text content must be substantial (not just headlines)
Clear category labels for classification
English language articles only
Manageable size for Google Colab free tier
Dataset Approval Process
Before using any Kaggle dataset, you MUST get instructor approval
Send email with: Dataset name and Kaggle URL, Number of articles and categories, Sample of 3-5 articles from the dataset, Justification for why this dataset is suitable
Wait for approval before proceeding - unapproved datasets will result in point deduction
📦 Deliverables
You must submit TWO deliverables for this project:

Deliverable 1: Individual GitHub Repository (Required)
Repository URL: Submit your individual GitHub portfolio repository link via Canvas
Project Folder: ITAI2373-NewsBot-Midterm within your portfolio repository
Content Requirements: Complete, runnable Jupyter notebook with all analyses
Well-organized code with clear documentation and comments
md in the project folder with project overview and your specific contributions
All required visualizations and analysis outputs
Clean folder structure with appropriate file organization
Note: Each team member submits their own repository link, but the project content should be identical across team members
Deliverable 2: Group Reflective Journal (Required)
File Name: NewsBot_Reflection_[TeamName].pdf
Length: Exactly 2 pages (no more, no less)
Submission: One submission per team via Canvas (designate one team member to submit)
Content Requirements: Team Collaboration Analysis, Technical Integration Challenges, Business Value Assessment, Individual Contributions, Future Enhancements, Professional Development Impact
Deliverable 3: Video Demonstration (Optional - 10 Bonus Points)
Length: 5-7 minutes maximum
Format: YouTube, Vimeo, or similar platform with public/unlisted access
Submission: Include link in your individual repository README
File Name Convention: MT_Video_Group_[Name/Number]_[Student_Full_Name]_ITAI2373
Detailed Video Content Requirements
Minute 1: Introduction and Team Overview - Introduce your team and the NewsBot Intelligence System project, briefly explain what your system does and why it's valuable, show your GitHub repository structure
Minutes 2-3: Live System Demonstration - MUST demonstrate with NEW articles (not from your training dataset), show the complete pipeline: input a news article → get classification, entities, sentiment, use at least 2 different article types
Minutes 4-5: Technical Deep Dive - Show 2-3 key visualizations from your analysis, explain one interesting finding or insight your system discovered, demonstrate one advanced feature
Minutes 6-7: Business Value and Conclusion - Explain real-world applications of your system, discuss what you learned and how this fits in your portfolio, mention future improvements or extensions
Video Technical Requirements
Screen Recording: Use tools like OBS, Zoom recording, or built-in screen capture
Audio Quality: Clear narration - use headset microphone if possible
Resolution: Minimum 720p, readable text and code
Platform: Upload to YouTube (unlisted), Vimeo, or Google Drive with public access
Accessibility: Ensure instructor can view without special permissions
 

🏆 Assessment Criteria (Total: 100 Points)
Your project will be evaluated based on technical implementation, system integration, and analytical depth.

Component

Points

Description

GitHub Repository

30

Well-organized code, documentation, and results

Technical Implementation

30

All modules integrated, working system

Analysis & Insights

30

Meaningful findings and business implications

Presentation & Documentation

10

Video demo + written report

File Naming Conventions (Required)
Report: MT_Report_Group_[Name/Number]_[Student_Full_Name]_ITAI2373
Video: MT_Video_Group_[Name/Number]_[Student_Full_Name]_ITAI2373 (it can be a page with the link to the video) 
Repository Folder: ITAI2373-NewsBot-Midterm 
⚠️REMEMBER: You MUST include a working link to your GitHub repository in the report.

🎯 Success Tips
For Computational Efficiency:
Start with small data samples for testing
Use efficient pandas operations and batch processing
Save intermediate results to avoid recomputation
Monitor Colab resource usage carefully
For Team Success:
Assign clear roles based on individual strengths
Use GitHub branches for parallel development
Schedule regular team check-ins
Document all decisions and challenges
For Technical Excellence:
Focus on code quality over complexity
Implement proper error handling
Create reusable, modular functions
Test with different data samples
 

🚀 Bonus Opportunities (Extra Credit)
Earn up to 15 additional points for:

Interactive Dashboard (5 pts): Streamlit/Gradio interface
Advanced Analysis (5 pts): Temporal trends, cross-lingual comparison
Custom Models (5 pts): Domain-specific NER training
Research Extension (10 pts): Novel approach with evaluation
 

⚠️ Important Notes
Academic Integrity
Collaboration within teams is expected and encouraged
AI-assisted coding is allowed to aid learning (not to code for you)
All external resources must be properly cited
Each team member must contribute meaningfully
Late Submission Policy
5% penalty per day late (after 48 hrs past due date)
Extensions only for documented emergencies
Technical Issues
Start early to identify and resolve computational limitations
Contact instructor immediately if dataset access issues arise
📞 Getting Help
Technical Questions:
Feel free to email me for urgent issues
Team Issues:
Contact instructor immediately for team conflicts
Individual contributions will be assessed via reflection papers
Alternative arrangements available if needed
🎓 Learning Outcomes
By completing this project, you will:

✅ Master NLP Pipeline Development from raw text to actionable insights
✅ Gain Professional Experience with industry-standard tools and practices
✅ Build Portfolio Content suitable for job applications and interviews
✅ Develop Collaboration Skills essential for software development careers
✅ Apply Business Thinking to connect technical results with real-world value
📋 Submission Checklist
Before submitting, ensure you have:

GitHub repository is public and well-organized
All code runs without errors in fresh Colab environment
Video presentation is accessible and within time limit 
Written report follows format guidelines
Individual reflections completed by all team members
All team members listed in Canvas submission
🎓 Final Thoughts
This project represents the culmination of your learning in the first half of our NLP course. It's designed to be challenging but achievable, comprehensive but focused, and most importantly, valuable for your professional development.

Remember that the goal isn't just to complete the assignment, but to build something you're genuinely proud to showcase. This NewsBot Intelligence System should demonstrate not only your technical skills but also your ability to think critically about real-world problems and communicate complex ideas effectively.

Learning to work with Kaggle datasets is an essential skill for any data scientist or NLP practitioner. The platform hosts thousands of datasets and competitions that will be valuable throughout your career.

Your success in this project will prepare you for the advanced topics in Modules 9-16 and provide a strong foundation for your capstone project. More importantly, it will give you concrete experience with the kind of NLP systems you'll build in your professional career.

Good luck, and remember: the best learning happens when you're building something meaningful!
