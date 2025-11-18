# 🚀 Almost-Million Velocity Rank

A Streamlit web app that finds YouTube videos approaching 1 million views and ranks them by **velocity** (views per day). Discover the next viral hit before it peaks!

## Features

- 🔍 **Search & Discover**: Find videos by topic using YouTube Data API v3
- 📊 **Velocity Ranking**: Sort videos by views per day to identify fast-growing content
- 🎯 **Smart Clustering**: ML-powered archetypes (Rocket 🚀, Steady 📊, Slow Grind 🐌)
- 📈 **Feature Importance**: Understand what drives velocity using Random Forest
- 🎬 **Watch In-App**: Embedded video player to watch directly
- ⚡ **Optimized**: Cached API calls and model training for speed

## Quick Start

### 1. Get a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable **YouTube Data API v3**
4. Create credentials → API Key
5. Copy your API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
YOUTUBE_API_KEY=your_actual_api_key_here
```

Alternatively, set as environment variable:

```bash
export YOUTUBE_API_KEY="your_api_key_here"
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Basic Workflow

1. **Enter Search Query**: Type a topic (e.g., "gaming", "cooking", "news")
2. **Set Parameters**: 
   - Max videos to fetch (50-500)
   - View count range (default: 750K-1M)
3. **Click "Fetch Videos"**: App will search and analyze
4. **Explore Results**: 
   - View velocity-ranked list
   - Filter by archetype
   - Watch videos in-app
   - Analyze clustering patterns

### Understanding the Metrics

- **Views**: Total view count
- **Velocity**: Views per day (primary ranking metric)
- **To 1M**: Views remaining to reach 1 million
- **Age**: Days since video was published

### Video Archetypes

The clustering algorithm groups videos into three types:

- **🚀 Rocket**: Young videos with explosive velocity
  - Just published, rapid growth
  - High views per day relative to age
  
- **📊 Steady**: Moderate age with consistent velocity
  - Established videos with steady momentum
  - Balanced growth pattern
  
- **🐌 Slow Grind**: Older videos still climbing
  - Published longer ago
  - Lower velocity but sustained growth

### Feature Importance

The Random Forest analysis shows which features correlate with higher velocity:

- **age_days**: Newer videos typically have higher velocity
- **like_ratio**: High engagement predicts faster growth
- **comment_ratio**: Discussion activity correlates with velocity

## Technical Details

### Architecture

```
app.py
├── YouTube API Integration
│   ├── search.list (find videos by query)
│   └── videos.list (get detailed statistics)
├── Feature Engineering
│   ├── Velocity calculation (views/day)
│   ├── Engagement ratios (likes/views, comments/views)
│   └── Distance metrics (to 1M)
├── Machine Learning
│   ├── K-Means clustering (3 archetypes)
│   └── Random Forest (feature importance)
└── Streamlit UI
    ├── Interactive controls
    ├── Velocity-ranked list
    ├── Embedded video player
    └── Visualizations (Plotly)
```

### API Usage

The app uses YouTube Data API v3 efficiently:

- **search.list**: Finds video IDs matching query
- **videos.list**: Fetches detailed statistics in batches of 50
- **Caching**: Results cached for 1 hour to minimize API calls
- **Rate Limits**: Respects YouTube API quotas

Default quota: 10,000 units/day
- search.list: 100 units per call
- videos.list: 1 unit per call
- Typical app usage: ~150-300 units per search

### ML Approach

**K-Means Clustering**
- Input features: age_days, views_per_day, like_ratio, comment_ratio
- Standardized before clustering
- 3 clusters with interpretable labels
- Deterministic (random_state=0)

**Random Forest Regression**
- Target: views_per_day
- Features: age_days, like_ratio, comment_ratio
- 100 trees, max_depth=5
- Purely explanatory (not used for ranking)

## Customization

### Adjust View Range

Default: 750K-999K views (videos close to 1M)

Modify via sidebar slider or in code:

```python
view_range = st.sidebar.slider(
    "View Count Filter",
    min_value=0,
    max_value=1_000_000,
    value=(750_000, 999_999),  # Change default here
    step=10_000
)
```

### Change Cluster Count

Default: 3 clusters (Rocket, Steady, Slow Grind)

Modify in `train_cluster_model()`:

```python
kmeans = KMeans(n_clusters=3, random_state=0)  # Change n_clusters
```

### Add More Features

To include additional features in ML models:

1. Compute in `compute_basic_features()`
2. Add to `feature_cols` in `train_cluster_model()`
3. Add to `feature_cols` in `compute_feature_importances()`

## Troubleshooting

### "YouTube API key not found"

**Solution**: Make sure `.env` file exists and contains:
```
YOUTUBE_API_KEY=your_key_here
```

### "No videos found"

**Possible causes**:
- Query too specific
- View range too narrow
- API quota exceeded

**Solutions**:
- Try broader search terms
- Widen view count range
- Check API quota in Google Cloud Console

### "Not enough videos for clustering"

**Solution**: Increase `max_results` or widen view count range

### API Rate Limits

If you hit quota limits:
- Results are cached for 1 hour
- Wait for quota reset (daily)
- Request quota increase in Google Cloud Console

## Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation
- **requests**: HTTP requests to YouTube API
- **scikit-learn**: ML models (KMeans, RandomForest)
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **python-dotenv**: Environment variable management

## Project Structure

```
.
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env.example       # API key template
├── .env               # Your API key (create this, not in git)
└── README.md          # This file
```

## API Key Security

⚠️ **Important**: Never commit your API key to version control!

Add to `.gitignore`:
```
.env
*.pyc
__pycache__/
.streamlit/secrets.toml
```

## Performance Tips

1. **Start Small**: Use 50-100 videos initially to test
2. **Cache Results**: App automatically caches for 1 hour
3. **Narrow Query**: Specific queries = faster, more relevant results
4. **Monitor Quota**: Check usage in Google Cloud Console

## Future Enhancements

Potential features to add:

- [ ] Export results to CSV
- [ ] Historical tracking (compare velocities over time)
- [ ] Multi-query comparison
- [ ] Custom date ranges
- [ ] More clustering algorithms
- [ ] Trend prediction models
- [ ] Social media sharing metrics

## License

MIT License - feel free to modify and use as needed!

## Credits

Built with:
- [Streamlit](https://streamlit.io/)
- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [scikit-learn](https://scikit-learn.org/)
- [Plotly](https://plotly.com/)

---

**Happy hunting for the next viral video! 🎯**
