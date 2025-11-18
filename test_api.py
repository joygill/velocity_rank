"""
Test your YouTube API key setup
Run this before using the main app to verify everything is configured correctly.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if YouTube API key is working correctly."""
    
    print("=" * 60)
    print("YouTube API Key Test")
    print("=" * 60)
    
    # Step 1: Check if API key is set
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        print("❌ FAILED: YOUTUBE_API_KEY not found in environment")
        print("\nTo fix:")
        print("1. Create a .env file in this directory")
        print("2. Add: YOUTUBE_API_KEY=your_key_here")
        print("\nOr set environment variable:")
        print("  export YOUTUBE_API_KEY='your_key_here'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Step 2: Test API key with a simple search
    print("Testing API key with YouTube search...")
    
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": "test",
                "type": "video",
                "maxResults": 1,
                "key": api_key
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                video_title = data["items"][0]["snippet"]["title"]
                print(f"✅ API key works! Found video: '{video_title}'")
                print()
                print("=" * 60)
                print("SUCCESS! Your API key is configured correctly.")
                print("You can now run: streamlit run app.py")
                print("=" * 60)
                return True
            else:
                print("⚠️  API returned empty results")
                return False
                
        elif response.status_code == 400:
            print("❌ FAILED: Bad Request (400)")
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                print(f"\nError: {error_msg}")
            except:
                print(f"\nRaw response: {response.text}")
            print("\nPossible causes:")
            print("- API key format is incorrect")
            print("- API key has invalid characters")
            return False
            
        elif response.status_code == 403:
            print("❌ FAILED: Forbidden (403)")
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                print(f"\nError: {error_msg}")
            except:
                print(f"\nRaw response: {response.text}")
                
            print("\nPossible causes:")
            print("1. YouTube Data API v3 not enabled")
            print("   → Go to: https://console.cloud.google.com/apis/library/youtube.googleapis.com")
            print("   → Click 'Enable'")
            print()
            print("2. API key restrictions blocking request")
            print("   → Go to: https://console.cloud.google.com/apis/credentials")
            print("   → Edit your API key")
            print("   → Under 'API restrictions', select 'YouTube Data API v3'")
            print()
            print("3. Billing not enabled (rare)")
            print("   → Go to: https://console.cloud.google.com/billing")
            return False
            
        elif response.status_code == 429:
            print("❌ FAILED: Quota Exceeded (429)")
            print("\nYou've hit the daily quota limit.")
            print("Wait until midnight Pacific Time for reset, or request quota increase.")
            return False
            
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"\nResponse: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ FAILED: Request timed out")
        print("\nCheck your internet connection")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Connection error")
        print("\nCheck your internet connection")
        return False
        
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    test_api_key()
