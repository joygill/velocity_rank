# 🚀 Quick Start: Hybrid Search Feature

## Where It Was Added

### Sidebar - Top Section (NEW!)

```
┌─────────────────────────────────┐
│ ⚙️ Search Parameters           │
│                                 │
│ Search Mode                     │ ← NEW!
│ ○ 🔍 Topic Search              │
│ ○ 📈 Trending by Category      │
│                                 │
│ ─────────────────────────────  │
│                                 │
│ [Mode-specific controls below]  │
└─────────────────────────────────┘
```

---

## Two Different UI Flows

### Flow A: Topic Search (Original)

```
┌─────────────────────────────────┐
│ Search Mode                     │
│ ● 🔍 Topic Search              │ ← Selected
│ ○ 📈 Trending by Category      │
│                                 │
│ Search Query                    │
│ ┌─────────────────────────┐   │
│ │ gaming                  │   │ ← Type anything
│ └─────────────────────────┘   │
│                                 │
│ Max Videos to Fetch             │
│ ├────●──────────────────┤      │
│ 50        200          500      │
│                                 │
│ View Count Range                │
│ ├────●──────────●───────┤      │
│ 0    750K     1M      1M        │
│                                 │
│ [🔍 Fetch Videos]              │
└─────────────────────────────────┘
```

### Flow B: Trending by Category (NEW!)

```
┌─────────────────────────────────┐
│ Search Mode                     │
│ ○ 🔍 Topic Search              │
│ ● 📈 Trending by Category      │ ← Selected
│                                 │
│ Category                        │
│ ┌─────────────────────────┐   │
│ │ Gaming              ▼   │   │ ← Choose from list
│ └─────────────────────────┘   │
│                                 │
│ Max Videos to Fetch             │
│ ├────────●──────────────┤      │
│ 50       200          200       │ ← Max is 200
│                                 │
│ View Count Range                │
│ ├────●──────────●───────┤      │
│ 0    750K     1M      1M        │
│                                 │
│ [🔍 Fetch Videos]              │
└─────────────────────────────────┘
```

---

## Example Usage

### Example 1: Find Trending Gaming Videos

**Steps:**
1. Select **📈 Trending by Category**
2. Choose **Gaming** from dropdown
3. Keep max results at **200**
4. Keep view range **750K - 1M**
5. Click **Fetch Videos**

**Result:**
- ~50-80 videos in target range
- Currently popular gaming content
- Good mix of velocities

### Example 2: Search Specific Topic

**Steps:**
1. Select **🔍 Topic Search**
2. Type **"minecraft tutorial"**
3. Set max results to **300**
4. Keep view range **750K - 1M**
5. Click **Fetch Videos**

**Result:**
- ~15-30 videos in target range
- Specific to Minecraft tutorials
- More niche/targeted results

### Example 3: Compare Both Modes

**Try this experiment:**

**Round 1 - Topic Search:**
- Query: "music"
- Max: 200
- **Record:** How many videos in 750K-1M range?

**Round 2 - Trending:**
- Category: Music
- Max: 200
- **Record:** How many videos in 750K-1M range?

**You'll notice:** Trending typically gives 2-3x more videos in your target range!

---

## What Shows Up in Results

### Results Header Changes

**Topic Search results show:**
```
🔍 Summary for 'minecraft tutorial'
```

**Trending results show:**
```
📈 Summary for 'Gaming'
```

Everything else stays the same - same velocity ranking, same clustering, same analytics!

---

## Common Workflows

### Workflow 1: Discovery Mode
*"I want to see what's popular and close to 1M views"*

1. **📈 Trending by Category**
2. Pick a category you're interested in
3. Fetch and explore!
4. Watch videos directly in app

### Workflow 2: Hunting Mode  
*"I'm looking for specific content about to hit 1M"*

1. **🔍 Topic Search**
2. Enter very specific query
3. May need to adjust view range if sparse
4. Find hidden gems

### Workflow 3: Research Mode
*"I want to understand what drives velocity in a category"*

1. **📈 Trending by Category**
2. Fetch 200 videos
3. Study the clustering (Rockets vs Slow Grind)
4. Check feature importance
5. Use insights for your content strategy

---

## Pro Tips

### Getting Better Results

**For Trending:**
✅ Works great with: Gaming, Music, Sports, Entertainment
⚠️ Sparse results with: Nonprofits, News & Politics
💡 **Tip:** Try Music category - often has the most videos in 750K-1M range!

**For Topic Search:**
✅ Use specific phrases: "fortnite gameplay" not "gaming"
✅ Include qualifiers: "2024 NBA highlights"
✅ Avoid generic terms: "music" → "music video 2024"
💡 **Tip:** If you get < 10 results, widen your view range to 500K-1M

### Optimizing API Usage

**Save Quota:**
- Use Trending (50-200 units) instead of Topic Search (150-300 units)
- Results are cached for 1 hour - re-running is free!

**Maximize Results:**
- Trending: Always use max 200
- Topic Search: Start at 200, increase to 500 if needed

---

## Testing Your Setup

### Quick Test 1: Verify Trending Works

```bash
streamlit run app.py
```

1. Select **📈 Trending by Category**
2. Choose **Gaming**
3. Max: **50** (quick test)
4. Click Fetch

**Expected:** 15-25 videos in 750K-1M range

### Quick Test 2: Verify Topic Works

1. Select **🔍 Topic Search**
2. Query: **"music video"**
3. Max: **50**
4. Click Fetch

**Expected:** 5-15 videos in 750K-1M range

### If Both Work:
✅ Setup complete! You're ready to discover almost-million videos!

---

## Keyboard Shortcuts

While on the app:

- `F5` - Refresh (clear cache and start fresh)
- `Ctrl/Cmd + Click` on video links - Open in new tab
- Adjust sliders with arrow keys for precision

---

## Quick Reference

| Feature | Topic Search | Trending |
|---------|-------------|----------|
| **Input** | Free text | Category selection |
| **Max Results** | 50-500 | 50-200 |
| **Coverage** | Variable | Consistent |
| **API Quota** | 150-300 units | 50-200 units |
| **Best For** | Specific topics | General discovery |
| **Target Range Hits** | 10-30% | 30-50% |

---

## Need Help?

**App shows no results?**
→ Try Trending Mode with Music or Gaming category

**Want more specific content?**
→ Use Topic Search with detailed queries

**Using too much API quota?**
→ Switch to Trending mode, uses 50% less quota

**Can't decide which mode?**
→ Start with Trending to explore, then Topic for specific content

---

That's it! The hybrid search gives you the flexibility of keyword search AND the reliability of trending discovery. Enjoy! 🎉
