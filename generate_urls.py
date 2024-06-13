import requests
from bs4 import BeautifulSoup

def generate_urls():
    base_urls = [
        "https://en.wikipedia.org/wiki/Category:Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Category:Machine_learning",
        "https://en.wikipedia.org/wiki/Category:Deep_learning",
        "https://en.wikipedia.org/wiki/Category:Computer_vision",
        "https://en.wikipedia.org/wiki/Category:Robotics",
        "https://en.wikipedia.org/wiki/Category:Data_science",
        "https://en.wikipedia.org/wiki/Category:Big_data",
        "https://en.wikipedia.org/wiki/Category:Internet_of_things",
        "https://en.wikipedia.org/wiki/Category:Cybersecurity"
    ]

    urls = set()

    for base_url in base_urls:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            url = link['href']
            if url.startswith('/wiki/') and ':' not in url:
                full_url = f"https://en.wikipedia.org{url}"
                urls.add(full_url)

    return list(urls)

if __name__ == "__main__":
    urls = generate_urls()
    with open('urls.txt', 'w') as f:
        for url in urls:
            f.write(f"{url}\n")
    print(f"Generated {len(urls)} URLs and saved to urls.txt")
