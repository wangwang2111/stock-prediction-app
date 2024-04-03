import streamlit as st
from st_functions import st_button, load_css
from PIL import Image
from constant import *
from streamlit_timeline import timeline
import plotly.graph_objects as go
import plotly.express as px

# icon = Image.open("favicon.jpg")
st.set_page_config(page_title="Profile", page_icon="📈", layout='wide')

load_css("main.css")

st.write("[![Star](https://img.shields.io/github/stars/dataprofessor/links.svg?logo=github&style=social)](https://github.com/wangwang2111/)")

col1, col2, col3, col4, col5 = st.columns(5)
col3.image(Image.open('dp.jpg'))

st.header(':violet[Dylan Nguyen]')
with st.expander('Connect with me'):
    with st.spinner(text="Loading details..."):
        # st.write()

        icon_size = 20

        # st_button('github', 'https://github.com/wangwang2111/', 'Read my Blogs', icon_size)
        st_button('twitter', 'hhttps://twitter.com/?lang=en', 'Follow me on Twitter', icon_size)
        st_button('linkedin', 'https://www.linkedin.com/in/dylan-nguyen-4b6a52287/', 'Follow me on LinkedIn', icon_size)
        st_button('instagram', 'https://www.instagram.com/wangg__wangg/', 'My profile on Instagram', icon_size)
        st_button('facebook', 'https://www.facebook.com/dangquang211102/', 'Check my profile on Facebook', icon_size)

st.subheader('About me', divider='violet')
with st.expander('About me'):
    with st.spinner(text="Loading details..."):
        st.write(info['Brief'])
        
st.subheader('Career snapshot', divider='violet')  
with st.spinner(text="Building line"):
    with open('timeline.json', "r") as f:
        data = f.read()
        timeline(data, height=450)


st.subheader('Skills & Tools ⚒️', divider='violet')
def skill_tab():
    rows,cols = len(info['skills'])//skill_col_size,skill_col_size
    skills = iter(info['skills'])
    if len(info['skills'])%skill_col_size!=0:
        rows+=1
    for x in range(rows):
        columns = st.columns(skill_col_size)
        for index_ in range(skill_col_size):
            try:
                columns[index_].button(next(skills))
            except:
                break
with st.spinner(text="Loading section..."):
    skill_tab()


st.subheader('Education 📖', divider='violet')

st.dataframe(info['edu'].set_index(info['edu'].columns[2]), use_container_width=True)
# fig = go.Figure(data=[go.Table(
#     header=dict(values=list(info['edu'].columns),
#                 fill_color='paleturquoise',
#                 align='left',height=50,font_size=16),
#     cells=dict(values=info['edu'].transpose().values.tolist(),
#                fill_color='lavender',
#                align='left',height=30,font_size=14))])

# fig.update_layout(width=900, height=500)
# st.plotly_chart(fig)
st.subheader('Projects 📚', divider='violet')
st.subheader('Research Papers 📝', divider='violet')

def plot_bar():
    st.info('Comparing Brute Force approach with the algorithms')
    temp1 = rapid_metrics.loc[['Brute-Force_Printed','printed'],:].reset_index().melt(id_vars=['category'],value_vars=['precision','recall','f1_score'],var_name='metrics',value_name='%').reset_index()
    
    temp2 = rapid_metrics.loc[['Brute-Force_Handwritten','handwritten'],:].reset_index().melt(id_vars=['category'],value_vars=['precision','recall','f1_score'],var_name='metrics',value_name='%').reset_index()
    
    cols = st.columns(2)
    
    fig = px.bar(temp1, x="metrics", y="%", 
             color="category", barmode = 'group')
     
    cols[0].plotly_chart(fig,use_container_width=True)
    
    fig = px.bar(temp2, x="metrics", y="%", 
             color="category", barmode = 'group')
    cols[1].plotly_chart(fig,use_container_width=True)
    
    

def image_and_status_loader(image_list,index=0):
    if index==0:
        img = Image.open(image_list[0]['path'])
        st.image(img,caption=image_list[0]['caption'],width=image_list[0]['width'])
       
    else:
        st.success('C-Cube algorithm for printed prescriptions')
        rapid_metrics.loc[['Brute-Force_Printed','printed'],:].plot(kind='bar')
        cols = st.columns(3)
        for index_,items in enumerate(image_list[0]):
            cols[index_].image(items['path'],caption=items['caption'],use_column_width=True)
     
        
        st.success('3 step filtering algorithm for handwritten algorithms')
        cols = st.columns(3)
        for index_,items in enumerate(image_list[1]):
            cols[index_].image(items['path'],caption=items['caption'],use_column_width=True)
        
        plot_bar()
        
        

def paper_summary(index):
    st.markdown('<h5><u><a style="color: black;" href="https://icpt.hust.edu.vn/en/proceedings/icpt-hust-2023/proceedings-of-icpt-hust-2023-12.html">' + paper_info['name'][index] + '</a></u></h5>', unsafe_allow_html=True)
    st.caption(paper_info['role'][index])
    st.caption(paper_info['journal'][index]+' , '+paper_info['publication'][index]+' , '+paper_info['year'][index])
    with st.expander('detailed description'):
        with st.spinner(text="Loading details..."):
            # st.write(paper_info['Summary'][index])
            pdfFileObj = open('pdfs/{}'.format(paper_info['file'][index]), 'rb')
            # image_and_status_loader(paper_info['images'][str(index)], index)
            # if index==0:
            #     rpa_metrics['time_improvement'] = rpa_metrics['non-ds']-rpa_metrics['ds']
            #     st.markdown('**Time taken per order involving Rx in seconds** (green indicates improvements from baseline)')
            #     cols = st.columns(3)
            #     for index_, row in rpa_metrics.iterrows():
            #         cols[index_].metric(row['category'],str(row['ds'])+'s',delta=str(round(row['time_improvement'],1))+'s' )
            st.download_button('download paper',pdfFileObj,file_name=paper_info['file'][index],mime='pdf')
    
paper_summary(0)

st.subheader('Achievements 🥇', divider='violet')
achievement_list = ''.join(['<li>'+item+'</li>' for item in info['achievements']])
st.markdown('<ul>'+achievement_list+'</ul>',unsafe_allow_html=True)
        
st.subheader('Daily routine as Data Analyst at BFChem', divider='violet')
graph = graphviz.Digraph()
graph.edge('8AM, start','check mails')
graph.edge('check mails','check metrics for pipelines')
graph.edge('check metrics for pipelines','resolve issues',label='If bug found')
graph.edge('check metrics for pipelines','10AM, check PowwerBI reports issues')
graph.edge('resolve issues','work on new PowerBI reports')
graph.edge('10AM, check PowwerBI reports issues','work on new PowerBI reports', label='Almost everyday')
graph.edge('work on new PowerBI reports','Python Webscraping to take data of the input market',label='Almost everyday')
graph.edge('Python Webscraping to take data of the input market', 'ML with NLP for accelerating cleaning process',label='seldom')
graph.edge('SQL Queries','EDA and cleaning the substances in the inputs market')
graph.edge('EDA and cleaning the substances in the inputs market','ML with NLP for accelerating cleaning process')
graph.edge('EDA and cleaning the substances in the inputs market','work on manager\'s assigned', label="if have")
graph.edge('work on manager\'s assigned','Some meeting')
graph.edge('work on manager\'s assigned','Around 4PM')
graph.edge('Around 4PM','end')
graph.edge('Some meeting','end',label='If too long')
graph.edge('Some meeting','work on manager\'s assigned')
st.graphviz_chart(graph)

pdfFileObj = open('pdfs/CV_NguyenDangQuang.pdf', 'rb')
st.sidebar.download_button('download resume',pdfFileObj,file_name='',mime='pdf')