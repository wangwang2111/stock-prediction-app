import pandas as pd
import graphviz as graphviz

edu = [['Bachelor Degree', 'Fintech', '7/2024', 'National Economics University of Vietnam', '3.75/4.0 GPA'],
       ['Certificate', 'The Data Science Course: Complete Data Science Bootcamp 2024', '1/2024','Udemy', '100%'],
       ['Certificate', 'Machine Learning A-Z: AI, Python & R + ChatGPT Prize [2024]', '2/2024', 'Udemy', '100%'],
       ['Certificate', 'Web Scraping in Python Selenium, Scrapy 2024', '3/2012', 'Udemy', '100%']]

info = {'name': 'Dylan Nguyen',
        'Brief': "With a passion for investing and data-driven decision-making, I pursued a Bachelor's degree in Financial Technology at the National Economics University (Vietnam), where I learned about financial big data analytics, portfolio management, risk management, and other fundamental financial areas.\n I applied my analytical skills and knowledge to various projects and work experiences, such as creating dashboards and reports on Sales, HRM, Inventory, and IT as a Data Analyst at BA DINH FOODSTUFF TECHNOLOGY JOINT STOCK COMPANY, and generating visual representations of sales and cost structure data using SQL and Power BI as a Data Analyst at VINATEX INTERNATIONAL JSC. Furthermore, I also applied machine learning (ANN, NLP, ...) into my workflow of cleaning and analyzing competitors, and import cost optimization in BA DINH FOODSTUFF.\n I also developed a website using HTML, CSS, JavaScript, and Django as a Web Developer at TRI NGHIA TECHNOLOGY TRADE AND SERVICES COMPANY LIMITED.\n My goal is to work in a leading financial institution or investment management firm as a Quantitative Analyst or Risk Manager, where I can leverage technology such as big data analytics, machine learning, and cloud computing to optimize investment portfolios and manage risks. I aspire to influence financial decision-making and have a positive impact on society.",
        'photo': {'path': 'dp.jpg', 'width': 150},
        'Mobile': '+846 921 5540',
        'Email': 'ndquang2111@gmail.com',
        'Medium': 'https://medium.com/@mehulgupta_7991/about',
        'City': 'Hanoi, Vietnam',
        'edu': pd.DataFrame(edu, columns=['Qualification', 'Stream', 'Year', 'Institute', 'Completion']),
        'skills': ['Data Science', 'Python', 'JavaScript', 'C++', 'R', 'SQL', 'M Query', 'DAX', 'PowerBI', 'Tableau', 'Selenium', 'Beautiful Soup', 'Streamlit', 'Tensorflow', 'pandas', 'matplotlib',
                   'seaborn', 'statsmodels', 'scikit learn', 'tensorflow', 'dplyr', 'ggplot2', 'tidymodels', 'fportfolio', 'HTML', 'CSS', 'Django', 'Flask'],
        'achievements': ['Academic Scholarship from National Economics University (2022)', 'Certificate of Silver Level in WorldQuant Brain Challenge (2023)', 'IELTS 7.5 (22/07/2023)', 'First runner-up Fintech Application in Banking Sector Competition -Attacker 2022 in Vietnam (12/02/2023)', 'Microsoft Office Specialist (MOS) 1976 pts (11/08/2023)'],
        'publication_url': 'https://icpt.hust.edu.vn/en/proceedings/icpt-hust-2023/proceedings-of-icpt-hust-2023-12.html'}

paper_info = {'name': ['Improve the quality of human resources in the banking sector in the context of digital transformation and climate change in Vietnam'],
              'publication': ['ICPT.HUST'
                              ],
              'journal': ['International Conference on Human Resources for Sustainable Development'],
              'year': ['2023'], 'role': ['Co-Author', 'Author'],
              'Summary': ['Based on the assessment of opportunities and challenges for human resources in the banking sector today in Vietnam, the authors propose recommendations to improve the quality of human resources in the Vietnamese banking sector, in the context of digital transformation and climate change, for the State Bank, commercial banks and higher education institutions.'],
              'file': ['HTQT-ICPT-HUST.pdf'],
              'images': {'0': [{'path': 'images/rpa1.PNG', 'caption': 'Digitization pipeline', 'width': 600}], '1': [[{'path': 'images/pr1.PNG', 'caption': 'Capture seed words'}, {'path': 'images/pr2.PNG', 'caption': 'cluster words using seed words'}, {'path': 'images/pr3.PNG', 'caption': 'clean junk words'}], [{'path': 'images/hw1.PNG', 'caption': 'Filter 1'}, {'path': 'images/hw2.PNG', 'caption': 'Filter 2'}, {'path': 'images/hw3.PNG', 'caption': 'Filter 3'}]]}}

models = ('Fashion MNIST samples using GAN', 'Cycle GAN for Image Translation')
cycle_models = ('Winter to Summer', 'Summer to Winter')
cycle_model_url = {cycle_models[0]: ['images/winter1.jpg', 'images/winter2.jpg', 'images/winter3.jpg'],
                   cycle_models[1]: ['images/summer1.jpg', 'images/summer2.jpg', 'images/summer3.jpg']}

rpa_metrics = pd.DataFrame([['Overall', 66.4, 72.5], ['printed rx', 54.6, 64.6], [
                           'handwritten', 67.3, 73.3]], columns=['category', 'ds', 'non-ds'])
rapid_metrics = pd.DataFrame([['printed', 91.6, 70, 79.4], ['handwritten', 21.1, 34.7, 26.2], [
                             'Brute-Force_Printed', 29.9, 82.7, 41.8], ['Brute-Force_Handwritten', 0.2, 62, 0.3]], columns=['category', 'precision', 'recall', 'f1_score'])
rapid_metrics = rapid_metrics.set_index(['category'])

skill_col_size = 5

books = {'amazon.com': 'https://www.amazon.com/LangChain-your-Pocket-Generative-Applications-ebook/dp/B0CTHQHT25',
         'gumroad': 'https://mehulgupta.gumroad.com/l/hmayz', 'amazon.in': 'https://www.amazon.in/dp/B0CTHQHT25'}
embed_component = {'linkedin': """<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
        <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="mehulgupta7991" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://in.linkedin.com/in/mehulgupta7991?trk=profile-badge"></a></div>""", 'medium': """<div style="overflow-y: scroll; height:500px;"> <div id="retainable-rss-embed" 
data-rss="https://medium.com/feed/retainable,https://medium.com/feed/data-science-in-your-pocket"
data-maxcols="3" 
data-layout="grid"
data-poststyle="inline" 
data-readmore="Read the rest" 
data-buttonclass="btn btn-primary" 
data-offset="0"></div></div> <script src="https://www.twilik.com/assets/retainable/rss-embed/retainable-rss-embed.js"></script>"""}

book_details = """<body>

<h3>Key Features:</h3>
<ul>
  <li>Step-by-step code explanations with expected outputs using GPT4 for each solution.</li>
  <li>No prerequisites: If you know Python, you can dive in.</li>
  <li>Practical, hands-on guide with minimal mathematical explanations.</li>
  <li>Apart from OpenAI API, it shows how to use local LLMs and Hugging Face API (free).</li>
  <li>The price is quite affordable compared to other similar books to make it accessible for all.</li>
</ul>

<h3>What It Covers:</h3>
<p>This beginner-friendly introduction covers:</p>
<ul>
  <li>Basics of Large Language Models (LLMs) and why LangChain is pivotal.</li>
  <li>Hello World tutorial for setting up LangChain and creating baseline applications.</li>
  <li>In-depth chapters on each LangChain module.</li>
  <li>Advanced problem-solving, including Multi-Document RAG, Hallucinations, NLP chains, and Evaluation for LLMs for supervised and unsupervised ML problems.</li>
  <li>Dedicated sections for Few-Shot Learning, Advanced Prompt Engineering using ReAct, Autonomous AI agents, and deployment using LangServe.</li>
</ul>

<h3>Who Should Read It?</h3>
<p>This book is for anyone keen on exploring AI, especially Generative AI. Whether you’re a Software Developer, Data Scientist, Student, or Content Writer, the focus on diverse use cases in LangChain and GenAI makes it equally valuable to all.</p>

<h3>Table of Contents</h3>
<ul>
<li>Introduction</li>
<li>Hello World</li>
<li>Different LangChain Modules</li>
  <li>Models & Prompts</li>
  <li>Chains</li>
  <li>Agents</li>
  <li>OutputParsers & Memory</li>
  <li>Callbacks</li>
  <li>RAG Framework & Vector Databases</li>
  <li>LangChain for NLP problems</li>
  <li>Handling LLM Hallucinations</li>
  <li>Evaluating LLMs</li>
  <li>Advanced Prompt Engineering</li>
  <li>Autonomous AI agents</li>
  <li>LangSmith & LangServe</li>
</ul>

</body>"""
