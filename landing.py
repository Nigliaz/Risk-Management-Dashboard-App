import numpy as np
from statistics import mean
from scipy.stats import percentileofscore
import streamlit as st
from scipy.stats import norm
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import requests
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder
from datetime import datetime, timedelta
from PIL import Image
import News
#Gets user input

def app():
    # Page Configuration

    image_path = 'media/Globe GPS CM - WHITE (1).png'
    image = Image.open(image_path)

    st.image(image)


 
    





