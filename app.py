"""
Almost-Million Velocity Rank - Streamlit App
===============================================
Find and rank YouTube videos approaching 1M views by their velocity (views per day).

Setup:
1. Set your YouTube API key as an environment variable:
   export YOUTUBE_API_KEY="your_api_key_here"
   
   Or create a .env file with:
   YOUTUBE_API_KEY=your_api_key_here

2. Install dependencies:
   pip install streamlit pandas requests scikit-learn python-dotenv

3. Run the app:
   streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
MIN_AGE_DAYS = 0.1  # Epsilon to avoid division by zero


# ============================================================================
# YOUTUBE API FUNCTIONS
# ============================================================================

# YouTube category IDs mapping
YOUTUBE_CATEGORIES = {
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Music": "10",
    "Pets & Animals": "15",
    "Sports": "17",
    "Gaming": "20",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "News & Politics": "25",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
    "Nonprofits & Activism": "29",
}

def get_api_key() -> str:
    """
    Retrieve YouTube API key from environment variable.
    
    Returns:
        str: API key
        
    Raises:
        ValueError: If API key is not set
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YouTube API key not found. Please set YOUTUBE_API_KEY environment variable."
        )
    return api_key


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_videos_from_youtube(query: str, max_results: int = 100) -> pd.DataFrame:
    """
    Fetch videos from YouTube Data API v3 based on a search query.
    
    Strategy:
    1. Use search.list to get video IDs matching the query
    2. Use videos.list to get detailed statistics and metadata
    
    Args:
        query: Search query (e.g., "music", "gaming")
        max_results: Maximum number of videos to fetch (max 500 per API design)
        
    Returns:
        DataFrame with video data including:
        - video_id, title, channel_title, published_at
        - view_count, like_count, comment_count
        - thumbnail_url, category_id
    """
    api_key = get_api_key()
    
    # Step 1: Search for videos
    # Note: We'll fetch more than needed to ensure we get enough in our target range
    search_results = []
    next_page_token = None
    
    # Calculate how many pages we need (max 50 results per page)
    pages_needed = (max_results + 49) // 50
    
    for _ in range(min(pages_needed, 10)):  # Limit to 10 pages (500 videos max)
        search_params = {
            "part": "id,snippet",
            "q": query,
            "type": "video",
            "maxResults": min(50, max_results),
            "key": api_key,
            "order": "relevance",  # relevance, date, rating, viewCount (viewCount requires chart parameter)
        }
        
        if next_page_token:
            search_params["pageToken"] = next_page_token
            
        search_response = requests.get(
            f"{YOUTUBE_API_BASE}/search",
            params=search_params
        )
        
        if search_response.status_code != 200:
            st.error(f"YouTube API error: {search_response.status_code}")
            try:
                error_data = search_response.json()
                if "error" in error_data:
                    error_message = error_data["error"].get("message", "Unknown error")
                    st.error(f"Error details: {error_message}")
                st.code(search_response.text, language="json")
            except:
                st.error(search_response.text)
            return pd.DataFrame()
            
        search_data = search_response.json()
        search_results.extend(search_data.get("items", []))
        
        next_page_token = search_data.get("nextPageToken")
        if not next_page_token:
            break
    
    if not search_results:
        st.warning("No videos found for this query.")
        return pd.DataFrame()
    
    # Extract video IDs
    video_ids = [item["id"]["videoId"] for item in search_results if "videoId" in item["id"]]
    
    # Step 2: Get detailed video statistics
    # API allows up to 50 IDs per request
    all_videos = []
    
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        
        video_params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(batch_ids),
            "key": api_key,
        }
        
        video_response = requests.get(
            f"{YOUTUBE_API_BASE}/videos",
            params=video_params
        )
        
        if video_response.status_code != 200:
            st.warning(f"Error fetching video details for batch {i//50 + 1}")
            continue
            
        video_data = video_response.json()
        all_videos.extend(video_data.get("items", []))
    
    # Step 3: Parse into DataFrame
    parsed_videos = []
    
    for video in all_videos:
        try:
            video_id = video["id"]
            snippet = video["snippet"]
            statistics = video.get("statistics", {})
            
            parsed_videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail_url": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                "category_id": snippet.get("categoryId", ""),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
            })
        except (KeyError, ValueError) as e:
            # Skip malformed videos
            continue
    
    df = pd.DataFrame(parsed_videos)
    
    if df.empty:
        st.warning("No valid video data could be parsed.")
        
    return df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_trending_videos_by_category(category_id: str, max_results: int = 200) -> pd.DataFrame:
    """
    Fetch trending/popular videos from a specific YouTube category.
    
    This uses the chart="mostPopular" parameter which gives better view count
    distribution compared to search queries. Great for finding videos in the
    "almost million" range.
    
    Args:
        category_id: YouTube category ID (e.g., "20" for Gaming)
        max_results: Maximum number of videos to fetch (max 200)
        
    Returns:
        DataFrame with video data (same format as fetch_videos_from_youtube)
    """
    api_key = get_api_key()
    
    all_videos = []
    next_page_token = None
    
    # Fetch popular videos in category
    # Note: chart=mostPopular is limited to 200 results total
    pages_needed = min((max_results + 49) // 50, 4)  # Max 4 pages = 200 videos
    
    for _ in range(pages_needed):
        video_params = {
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "videoCategoryId": category_id,
            "maxResults": min(50, max_results),
            "key": api_key,
            "regionCode": "US",  # Can be changed to other regions
        }
        
        if next_page_token:
            video_params["pageToken"] = next_page_token
        
        video_response = requests.get(
            f"{YOUTUBE_API_BASE}/videos",
            params=video_params
        )
        
        if video_response.status_code != 200:
            st.error(f"YouTube API error: {video_response.status_code}")
            try:
                error_data = video_response.json()
                if "error" in error_data:
                    error_message = error_data["error"].get("message", "Unknown error")
                    st.error(f"Error details: {error_message}")
                st.code(video_response.text, language="json")
            except:
                st.error(video_response.text)
            return pd.DataFrame()
        
        video_data = video_response.json()
        all_videos.extend(video_data.get("items", []))
        
        next_page_token = video_data.get("nextPageToken")
        if not next_page_token:
            break
    
    if not all_videos:
        st.warning("No trending videos found for this category.")
        return pd.DataFrame()
    
    # Parse into DataFrame (same format as search results)
    parsed_videos = []
    
    for video in all_videos:
        try:
            video_id = video["id"]
            snippet = video["snippet"]
            statistics = video.get("statistics", {})
            
            parsed_videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail_url": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                "category_id": snippet.get("categoryId", ""),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
            })
        except (KeyError, ValueError) as e:
            # Skip malformed videos
            continue
    
    df = pd.DataFrame(parsed_videos)
    
    if df.empty:
        st.warning("No valid video data could be parsed.")
    
    return df



# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_basic_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity and engagement features for videos.
    
    Features computed:
    - age_days: Days since video was published
    - views_per_day: Velocity metric (views / age)
    - distance_to_1M: How far from 1 million views
    - like_ratio: Likes per view
    - comment_ratio: Comments per view
    
    Args:
        df_raw: Raw DataFrame from YouTube API
        
    Returns:
        DataFrame with additional computed features
    """
    df = df_raw.copy()
    
    # Parse published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    
    # Calculate age in days
    now = datetime.now(timezone.utc)
    df["age_days"] = (now - df["published_at"]).dt.total_seconds() / 86400
    
    # Ensure minimum age to avoid division by zero
    df["age_days"] = df["age_days"].clip(lower=MIN_AGE_DAYS)
    
    # Calculate velocity (views per day)
    df["views_per_day"] = df["view_count"] / df["age_days"]
    
    # Calculate distance to 1 million
    df["distance_to_1M"] = 1_000_000 - df["view_count"]
    
    # Calculate engagement ratios (safe division)
    df["like_ratio"] = df.apply(
        lambda row: row["like_count"] / row["view_count"] if row["view_count"] > 0 else 0,
        axis=1
    )
    
    df["comment_ratio"] = df.apply(
        lambda row: row["comment_count"] / row["view_count"] if row["view_count"] > 0 else 0,
        axis=1
    )
    
    # Create YouTube watch URL
    df["youtube_url"] = df["video_id"].apply(
        lambda vid: f"https://www.youtube.com/watch?v={vid}"
    )
    
    return df


def filter_near_million(
    df: pd.DataFrame, 
    min_views: int = 750_000, 
    max_views: int = 1_000_000
) -> pd.DataFrame:
    """
    Filter videos to those within specified view count range.
    
    Args:
        df: DataFrame with video data
        min_views: Minimum view count (inclusive)
        max_views: Maximum view count (exclusive)
        
    Returns:
        Filtered DataFrame
    """
    mask = (df["view_count"] >= min_views) & (df["view_count"] < max_views)
    return df[mask].copy()


# ============================================================================
# MACHINE LEARNING
# ============================================================================

@st.cache_resource
def train_cluster_model(
    df_filtered: pd.DataFrame, 
    n_clusters: int = 3
) -> Tuple[KMeans, pd.DataFrame]:
    """
    Train k-means clustering model to identify video archetypes.
    
    Clusters represent different video trajectories:
    - "Rocket": New videos with high velocity
    - "Steady": Middle-aged videos with moderate velocity
    - "Slow Grind": Older videos with low velocity
    
    Args:
        df_filtered: DataFrame with computed features
        n_clusters: Number of clusters (default 3)
        
    Returns:
        Tuple of (trained KMeans model, DataFrame with cluster assignments)
    """
    if len(df_filtered) < n_clusters:
        st.warning(f"Not enough videos ({len(df_filtered)}) for {n_clusters} clusters.")
        df_filtered["cluster_label"] = "N/A"
        return None, df_filtered
    
    # Select features for clustering
    feature_cols = ["age_days", "views_per_day", "like_ratio", "comment_ratio"]
    X = df_filtered[feature_cols].copy()
    
    # Handle any missing or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Standardize features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    
    # Add cluster IDs to dataframe
    df_clustered = df_filtered.copy()
    df_clustered["cluster_id"] = cluster_ids
    
    # Compute cluster characteristics for labeling
    cluster_stats = df_clustered.groupby("cluster_id").agg({
        "age_days": "mean",
        "views_per_day": "mean",
    }).reset_index()
    
    # Sort by velocity (descending) and age (ascending) to assign labels
    cluster_stats["velocity_rank"] = cluster_stats["views_per_day"].rank(ascending=False)
    cluster_stats["age_rank"] = cluster_stats["age_days"].rank(ascending=True)
    
    # Assign human-readable labels
    # Highest velocity + youngest = Rocket
    # Middle = Steady
    # Lowest velocity + oldest = Slow Grind
    def assign_label(row):
        if row["velocity_rank"] == 1:
            return "🚀 Rocket"
        elif row["velocity_rank"] == n_clusters:
            return "🐌 Slow Grind"
        else:
            return "📊 Steady"
    
    cluster_stats["cluster_label"] = cluster_stats.apply(assign_label, axis=1)
    
    # Map labels back to main dataframe
    label_map = dict(zip(cluster_stats["cluster_id"], cluster_stats["cluster_label"]))
    df_clustered["cluster_label"] = df_clustered["cluster_id"].map(label_map)
    
    return kmeans, df_clustered


@st.cache_resource
def compute_feature_importances(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Use Random Forest to understand which features correlate with velocity.
    
    This is purely explanatory - we're trying to understand what drives
    high velocity, not predict it for new videos.
    
    Args:
        df_filtered: DataFrame with computed features
        
    Returns:
        DataFrame with feature names and their importance scores
    """
    if len(df_filtered) < 10:
        st.warning("Not enough videos for feature importance analysis.")
        return pd.DataFrame()
    
    # Select features and target
    feature_cols = ["age_days", "like_ratio", "comment_ratio"]
    X = df_filtered[feature_cols].copy()
    y = df_filtered["views_per_day"].copy()
    
    # Handle any missing or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.fillna(0)
    
    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=0,
        max_depth=5,
        min_samples_split=5
    )
    rf.fit(X, y)
    
    # Extract feature importances
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return importance_df


# ============================================================================
# STREAMLIT UI
# ============================================================================

def format_number(num: float) -> str:
    """Format large numbers with K/M suffixes."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"


def main():
    """Main Streamlit app."""
    
    st.set_page_config(
        page_title="Almost-Million Velocity Rank",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Almost-Million Velocity Rank")
    st.markdown("""
    Discover YouTube videos racing toward 1 million views and rank them by **velocity** 
    (views per day). Find the next viral hit before it peaks!
    """)
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    st.sidebar.header("⚙️ Search Parameters")
    
    # Search mode selector
    search_mode = st.sidebar.radio(
        "Search Mode",
        ["🔍 Topic Search", "📈 Trending by Category"],
        help="Topic Search: Search by keywords (flexible but may be sparse)\n"
             "Trending: Get popular videos in a category (better coverage)"
    )
    
    # Show appropriate controls based on mode
    if search_mode == "🔍 Topic Search":
        query = st.sidebar.text_input(
            "Search Query",
            value="music",
            help="Enter a topic to search for (e.g., 'gaming', 'cooking', 'news')"
        )
        category_id = None
        search_label = query
    else:  # Trending by Category
        category_name = st.sidebar.selectbox(
            "Category",
            options=list(YOUTUBE_CATEGORIES.keys()),
            index=3,  # Default to Gaming
            help="Select a YouTube category to find trending videos"
        )
        category_id = YOUTUBE_CATEGORIES[category_name]
        query = None
        search_label = category_name
    
    # Max results slider - adjusted based on mode
    if search_mode == "🔍 Topic Search":
        max_results = st.sidebar.slider(
            "Max Videos to Fetch",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="More videos = better results but slower API calls"
        )
    else:  # Trending mode has a 200 video limit
        max_results = st.sidebar.slider(
            "Max Videos to Fetch",
            min_value=50,
            max_value=200,
            value=200,
            step=50,
            help="Trending API limited to 200 videos max"
        )
    
    st.sidebar.subheader("View Count Range")
    
    view_range = st.sidebar.slider(
        "View Count Filter",
        min_value=0,
        max_value=1_000_000,
        value=(750_000, 999_999),
        step=10_000,
        format="%d",
        help="Only show videos within this view count range"
    )
    min_views, max_views = view_range
    
    fetch_button = st.sidebar.button("🔍 Fetch Videos", type="primary")
    
    # ========================================================================
    # FETCH AND PROCESS DATA
    # ========================================================================
    
    if not fetch_button and "df_filtered" not in st.session_state:
        st.info("👈 Set your parameters and click **Fetch Videos** to get started!")
        st.markdown("---")
        st.markdown("### 📋 How it works:")
        st.markdown("""
        **Choose Your Search Mode:**
        
        - **🔍 Topic Search**: Search by any keyword or phrase (e.g., "gaming", "cooking tutorial")
          - Flexible and customizable
          - May have sparse results in target view range
          
        - **📈 Trending by Category**: Get popular videos from YouTube categories
          - Better view count distribution
          - More videos in your target range
          - Great for discovering what's currently popular
        
        **Then:**
        1. **Filter**: Select videos close to 1M views (default: 750K-1M)
        2. **Rank**: Sort by velocity (views per day)
        3. **Analyze**: See clustering patterns and what drives velocity
        4. **Watch**: Click any video to watch it directly in the app!
        """)
        return
    
    if fetch_button:
        if search_mode == "🔍 Topic Search":
            with st.spinner(f"🔎 Searching YouTube for '{search_label}'..."):
                df_raw = fetch_videos_from_youtube(query, max_results)
        else:  # Trending by Category
            with st.spinner(f"📈 Fetching trending {search_label} videos..."):
                df_raw = fetch_trending_videos_by_category(category_id, max_results)
        
        if df_raw.empty:
            st.error("No videos found. Try different parameters.")
            return
        
        # Compute features
        df_features = compute_basic_features(df_raw)
        
        # Filter to near-million range
        df_filtered = filter_near_million(df_features, min_views, max_views)
        
        if df_filtered.empty:
            st.warning(f"No videos found in range {format_number(min_views)}-{format_number(max_views)} views.")
            st.info(f"Found {len(df_features)} total videos. Try adjusting the view count range.")
            return
        
        # Sort by velocity
        df_filtered = df_filtered.sort_values("views_per_day", ascending=False).reset_index(drop=True)
        
        # Train clustering model
        with st.spinner("🤖 Training clustering model..."):
            kmeans, df_clustered = train_cluster_model(df_filtered, n_clusters=3)
            
        # Compute feature importances
        with st.spinner("📊 Computing feature importances..."):
            importance_df = compute_feature_importances(df_filtered)
        
        # Store in session state
        st.session_state["df_filtered"] = df_clustered
        st.session_state["importance_df"] = importance_df
        st.session_state["search_label"] = search_label
        st.session_state["search_mode"] = search_mode
    
    # Retrieve from session state
    df_filtered = st.session_state.get("df_filtered")
    importance_df = st.session_state.get("importance_df")
    search_label = st.session_state.get("search_label", "")
    current_search_mode = st.session_state.get("search_mode", "🔍 Topic Search")
    
    if df_filtered is None or df_filtered.empty:
        return
    
    # ========================================================================
    # SUMMARY METRICS
    # ========================================================================
    
    st.markdown("---")
    mode_emoji = "🔍" if current_search_mode == "🔍 Topic Search" else "📈"
    st.subheader(f"{mode_emoji} Summary for '{search_label}'")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Videos Found", len(df_filtered))
    
    with col2:
        median_velocity = df_filtered["views_per_day"].median()
        st.metric("Median Velocity", f"{format_number(median_velocity)}/day")
    
    with col3:
        avg_distance = df_filtered["distance_to_1M"].mean()
        st.metric("Avg Distance to 1M", format_number(avg_distance))
    
    with col4:
        avg_age = df_filtered["age_days"].mean()
        st.metric("Avg Age", f"{avg_age:.0f} days")
    
    # ========================================================================
    # CLUSTER FILTER
    # ========================================================================
    
    if "cluster_label" in df_filtered.columns:
        st.markdown("---")
        cluster_options = ["All"] + sorted(df_filtered["cluster_label"].unique().tolist())
        selected_cluster = st.selectbox(
            "Filter by Archetype",
            options=cluster_options,
            help="Filter videos by their cluster archetype"
        )
        
        if selected_cluster != "All":
            df_display = df_filtered[df_filtered["cluster_label"] == selected_cluster].copy()
        else:
            df_display = df_filtered.copy()
    else:
        df_display = df_filtered.copy()
    
    # ========================================================================
    # VELOCITY-RANKED VIDEO LIST
    # ========================================================================
    
    st.markdown("---")
    st.subheader(f"🏆 Top Videos by Velocity ({len(df_display)} shown)")
    
    for idx, row in df_display.head(20).iterrows():
        with st.container():
            col_img, col_info = st.columns([1, 3])
            
            with col_img:
                if row["thumbnail_url"]:
                    st.image(row["thumbnail_url"], use_container_width=True)
            
            with col_info:
                # Title with link
                st.markdown(f"### [{row['title']}]({row['youtube_url']})")
                
                # Channel and cluster badge
                badge = row.get("cluster_label", "")
                st.markdown(f"**{row['channel_title']}** {badge if badge else ''}")
                
                # Metrics row
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Views", format_number(row["view_count"]))
                
                with metric_col2:
                    st.metric("Velocity", f"{format_number(row['views_per_day'])}/day")
                
                with metric_col3:
                    st.metric("To 1M", format_number(row["distance_to_1M"]))
                
                # Additional info
                st.caption(f"📅 Published {row['age_days']:.0f} days ago | "
                          f"👍 {format_number(row['like_count'])} likes | "
                          f"💬 {format_number(row['comment_count'])} comments")
                
                # Embedded video player
                with st.expander("▶️ Watch Video"):
                    st.video(row["youtube_url"])
            
            st.markdown("---")
    
    if len(df_display) > 20:
        st.info(f"Showing top 20 of {len(df_display)} videos. Adjust filters to see more.")
    
    # ========================================================================
    # CLUSTERING VISUALIZATION
    # ========================================================================
    
    if "cluster_label" in df_filtered.columns:
        st.markdown("---")
        st.subheader("🎯 Video Archetypes (Clustering)")
        
        st.markdown("""
        Videos are grouped into three archetypes based on their age, velocity, 
        and engagement patterns:
        
        - **🚀 Rocket**: Young videos with explosive velocity
        - **📊 Steady**: Moderate age and consistent velocity
        - **🐌 Slow Grind**: Older videos still climbing slowly
        """)
        
        # Create scatter plot
        fig = px.scatter(
            df_filtered,
            x="age_days",
            y="views_per_day",
            color="cluster_label",
            hover_data=["title", "view_count", "channel_title"],
            labels={
                "age_days": "Age (days)",
                "views_per_day": "Velocity (views/day)",
                "cluster_label": "Archetype"
            },
            title="Video Archetypes: Age vs Velocity"
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    
    if importance_df is not None and not importance_df.empty:
        st.markdown("---")
        st.subheader("🔍 What Drives Velocity?")
        
        st.markdown("""
        Using Random Forest regression, we can understand which features are most 
        associated with higher velocity. This is purely explanatory - it shows 
        correlations, not causation.
        """)
        
        # Create bar chart
        fig = go.Figure([go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation='h',
            marker_color='indianred'
        )])
        
        fig.update_layout(
            title="Feature Importance for Velocity Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        top_feature = importance_df.iloc[0]["feature"]
        interpretation_map = {
            "age_days": "**Age** is the strongest predictor - newer videos tend to have higher velocity as they're in their viral growth phase.",
            "like_ratio": "**Engagement (likes)** drives velocity - videos with higher like ratios tend to grow faster.",
            "comment_ratio": "**Discussion (comments)** drives velocity - videos sparking conversation tend to grow faster."
        }
        
        st.info(interpretation_map.get(top_feature, "Feature importance analysis complete."))
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.caption("Built with Streamlit • Data from YouTube Data API v3 • ML powered by scikit-learn")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        st.error(str(e))
        st.info("Please set your YouTube API key. See instructions at the top of app.py")
