import streamlit as st
import requests
import pandas as pd

# --- Function to call NewsAPI ---
def get_news(api_key, query):
    """
    Fetches news from NewsAPI based on a query.
    """
    url = "https://newsapi.org/v2/everything"
    
    # Parameters for the API call
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevancy', # You can change this to 'publishedAt'
        'pageSize': 100,      # Max 100
    }
    
    # Headers with your API key
    headers = {
        'X-Api-Key': api_key
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        # Raise an exception if the request was unsuccessful
        response.raise_for_status() 
        
        data = response.json()
        
        # Process the articles
        articles = data.get('articles', [])
        
        # Extract only the author and URL
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
        # Handle HTTP errors (like 401 Unauthorized, 429 Too Many Requests)
        st.error(f"API Error: {err.response.json().get('message')}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📰 Journalist & Article Finder (via NewsAPI)")
st.markdown("This tool searches NewsAPI for a topic and returns a list of authors and article URLs.")

# --- SIDEBAR for Inputs ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # We use st.secrets for the deployed app, but allow text_input for local testing
    # This is the "Pro" way to handle keys in deployed apps
    try:
        # Try to get the key from Streamlit Secrets
        default_key = st.secrets["NEWS_API_KEY"]
    except:
        # If secrets don't exist (local testing), set to empty
        default_key = ""

    api_key = st.text_input("Enter your NewsAPI Key", type="password", value=default_key)
    query = st.text_input("Enter search query (e.g., 'real estate')", "real estate")
    
    search_button = st.button("Search for Articles")

# --- MAIN CONTENT Area for Results ---
if search_button:
    if not api_key:
        st.warning("Please enter your NewsAPI Key in the sidebar to proceed.")
    elif not query:
        st.warning("Please enter a search query in the sidebar.")
    else:
        with st.spinner(f"Searching for articles about '{query}'..."):
            results = get_news(api_key, query)
            
            if results:
                st.success(f"Found {len(results)} articles!")
                
                # Create a DataFrame for better display
                df = pd.DataFrame(results)
                
                # Clean up data: remove rows where author is None or empty
                df_cleaned = df.dropna(subset=['Author'])
                df_cleaned = df_cleaned[df_cleaned['Author'] != '']
                
                st.info(f"Showing {len(df_cleaned)} articles with a listed author.")
                
                # Display the data in an interactive table
                st.dataframe(df_cleaned)
                
                # Add a download button for the team
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
