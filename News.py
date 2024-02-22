import streamlit as st
from pyfinviz.news import News

def app():
    # Sidebar refresh button
    if st.sidebar.button("Refresh"):
        # Use Streamlit's session state to trigger a rerun when the button is clicked
        st.session_state['refresh'] = not st.session_state.get('refresh', False)

    # Initialize and fetch the data
    news = News()

    # Display URL
    st.write("Scraped URL:", news.main_url)

    st.subheader("News DataFrame")
    st.dataframe(news.news_df, width=1600, height=800)  # Adjust width and height as needed

    # Display the Blogs DataFrame
    st.subheader("Blogs DataFrame")
    st.dataframe(news.blogs_df, width=1600, height=800)  # Adjust width and height as needed




# Ensure that the app function is called to display the app
if __name__ == "__main__":
    app()