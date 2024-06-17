import requests

def check_internet_connectivity():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("Internet connectivity check passed.")
        else:
            print("Internet connectivity check failed.")
    except requests.ConnectionError:
        print("No internet connection available.")
    except requests.Timeout:
        print("The request timed out.")

if __name__ == "__main__":
    check_internet_connectivity()
