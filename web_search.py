import requests

def perform_web_search(query):
    # Define the search engine API URL and parameters
    api_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": "YOUR_SEARCH_ENGINE_ID",  # Replace with your search engine ID
        "key": "YOUR_API_KEY"  # Replace with your API key
    }

    # Send the request to the search engine API
    response = requests.get(api_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        search_results = response.json()
        return search_results
    else:
        print(f"Error: {response.status_code}")
        return None

def process_search_results(search_results):
    # Extract relevant information from the search results
    processed_results = []
    if "items" in search_results:
        for item in search_results["items"]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            processed_results.append({"title": title, "snippet": snippet, "link": link})
    return processed_results

if __name__ == "__main__":
    query = "example query"
    search_results = perform_web_search(query)
    if search_results:
        processed_results = process_search_results(search_results)
        for result in processed_results:
            print(f"Title: {result['title']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Link: {result['link']}")
            print()
