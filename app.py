import streamlit as st
import requests
import pandas as pd

# --- Function to call NewsAPI (Same as before) ---
def get_news(api_key, query):
    """
    Fetches news from NewsAPI based on a query.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100,
    }
    headers = {'X-Api-Key': api_key}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() 
        data = response.json()
        articles = data.get('articles', [])
        
        journalist_list = []
        for article in articles:
            journalist_list.append({
                'Author': article.get('author'),
                'Article URL': article.get('url'),
                'Title': article.get('title'),
                'Source': article.get('source', {}).get('name')
            })
        return journalist_list

    except requests.exceptions.HTTPError as err:
        st.error(f"API Error: {err.response.json().get('message')}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📰 Journalist & Article Finder")
st.markdown("This tool searches NewsAPI for a topic. Please use your own NewsAPI key.")


# --- SIDEBAR for Inputs (Edited Section) ---
st.sidebar.header("⚙️ Settings")

# NEW INSTRUCTIONS ADDED HERE
st.sidebar.markdown(
    """
    **1. Get Your API Key**
    Register as an individual here to generate your key: 
    [https://newsapi.org/register](https://newsapi.org/register)
    
    **2. Enter Key Below**
    """
)
# --- API KEY INPUT: Always visible for user entry ---
api_key = st.sidebar.text_input("Enter YOUR NewsAPI Key", type="password")

query = st.sidebar.text_input("Enter search query (e.g., 'real estate')", "real estate")
search_button = st.sidebar.button("Search for Articles")


# --- MAIN CONTENT Area for Results ---
if search_button:
    # Check if the API key is available
    if not api_key:
        st.error("Please enter your NewsAPI Key in the sidebar to run the search.")
    elif not query:
        st.warning("Please enter a search query.")
    else:
        with st.spinner(f"Searching for articles about '{query}'..."):
            results = get_news(api_key, query)
            
            if results:
                st.success(f"Found {len(results)} articles!")
                df = pd.DataFrame(results)
                df_cleaned = df.dropna(subset=['Author'])
                df_cleaned = df_cleaned[df_cleaned['Author'] != '']
                
                st.info(f"Showing {len(df_cleaned)} articles with a listed author.")
                st.dataframe(df_cleaned)
                
                # Download button
                @st.cache_data
                def convert_df(df):
                   return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(df_cleaned)
                st.download_button(
                   label="Download results as CSV",
                   data=csv,
                   file_name=f'{query}_journalists.csv',
                   mime='text/csv',
                )
            else:
                st.warning("No results found or an error occurred.")
