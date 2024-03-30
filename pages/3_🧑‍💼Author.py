import streamlit as st
from st_functions import st_button, load_css
from PIL import Image

# icon = Image.open("favicon.jpg")
st.set_page_config(page_title="Profile", page_icon="📈")

load_css("style.css")

st.title('Profile')

st.write("[![Star](https://img.shields.io/github/stars/dataprofessor/links.svg?logo=github&style=social)](https://github.com/wangwang2111/)")

col1, col2, col3, col4, col5 = st.columns(5)
col3.image(Image.open('dp.jpg'))

col1, col2, col3 = st.columns(3)

st.header('Dylan Nguyen')

st.write('Data Scientist, ')

icon_size = 20

# st_button('github', 'https://github.com/wangwang2111/', 'Read my Blogs', icon_size)
st_button('twitter', 'hhttps://twitter.com/?lang=en', 'Follow me on Twitter', icon_size)
st_button('linkedin', 'https://www.linkedin.com/in/dylan-nguyen-4b6a52287/', 'Follow me on LinkedIn', icon_size)
st_button('instagram', 'https://www.instagram.com/wangg__wangg/', 'My profile on Instagram', icon_size)
st_button('facebook', 'https://www.facebook.com/dangquang211102/', 'Check my profile on Facebook', icon_size)
