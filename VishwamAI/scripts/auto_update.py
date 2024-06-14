import subprocess
import sys
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='/home/ubuntu/home/ubuntu/VishwamAI/logs/auto_update.log', level=logging.INFO)

def verify_package_signature(package_name):
    try:
        # Verify the package signature using GPG
        subprocess.check_call(["gpg", "--verify", f"{package_name}.sig", package_name])
        logging.info(f"{datetime.now()} - Package {package_name} signature verified successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{datetime.now()} - Package {package_name} signature verification failed: {e}")
        return False

def auto_update():
    try:
        # Check for internet connectivity
        subprocess.check_call(["ping", "-c", "1", "8.8.8.8"])
        logging.info(f"{datetime.now()} - Internet connectivity check passed.")

        # Update pip itself
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        logging.info(f"{datetime.now()} - Pip updated successfully.")

        # List outdated packages
        outdated_packages = subprocess.check_output([sys.executable, "-m", "pip", "list", "--outdated"]).decode("utf-8")

        # Parse the list of outdated packages
        packages_to_update = []
        for line in outdated_packages.splitlines()[2:]:
            package_name = line.split()[0]
            packages_to_update.append(package_name)

        # Update each outdated package
        for package in packages_to_update:
            try:
                if verify_package_signature(package):
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                    logging.info(f"{datetime.now()} - Package {package} updated successfully.")
                else:
                    logging.warning(f"{datetime.now()} - Skipping update for package {package} due to failed signature verification.")
            except subprocess.CalledProcessError as e:
                logging.error(f"{datetime.now()} - An error occurred while updating package {package}: {e}")

        print("All Python packages are up to date.")
        logging.info(f"{datetime.now()} - All Python packages are up to date.")

        # Update system packages
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call(["sudo", "DEBIAN_FRONTEND=noninteractive", "apt-get", "upgrade", "-y"])
        logging.info(f"{datetime.now()} - All system packages are up to date.")

        # Check for new versions of VishwamAI or related data/models
        # Example: Pull the latest changes from a GitHub repository
        repo_path = "/home/ubuntu/home/ubuntu/VishwamAI"
        if os.path.exists(repo_path):
            subprocess.check_call(["git", "-C", repo_path, "pull"])
            logging.info(f"{datetime.now()} - VishwamAI repository is up to date.")

        print("VishwamAI repository is up to date.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{datetime.now()} - An error occurred while updating: {e}")
        print(f"An error occurred while updating: {e}")
    except subprocess.TimeoutExpired as e:
        logging.error(f"{datetime.now()} - Internet connectivity check failed: {e}")
        print(f"Internet connectivity check failed: {e}")

if __name__ == "__main__":
    auto_update()
