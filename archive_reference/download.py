import requests
import os

def download_github_file(user, repo, file_path_in_repo, branch='main', local_filename=None):
    """
    Downloads a specific file from a public GitHub repository.

    Args:
        user (str): The GitHub username or organization (e.g., 'octocat').
        repo (str): The repository name (e.g., 'Spoon-Knife').
        file_path_in_repo (str): The full path to the file within the repository
                                 (e.g., 'data/my_data.json' or 'src/utils/helper.py').
        branch (str): The branch of the repository (default is 'main').
        local_filename (str, optional): The name to save the file as locally.
                                        If None, it uses the original filename from file_path_in_repo.
    """
    # Construct the raw GitHub URL for the file
    # This URL directly points to the raw content of the file
    raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path_in_repo}"
    
    # Determine the name for the local file
    if local_filename is None:
        # If no local filename is provided, use the base name of the file from the repo path
        local_filename = os.path.basename(file_path_in_repo)

    print(f"Attempting to download file from: {raw_url}")
    print(f"Saving as: {local_filename}")
    
    try:
        # Send an HTTP GET request to the raw file URL
        response = requests.get(raw_url)
        
        # Raise an HTTPError for bad responses (4xx or 5xx status codes)
        response.raise_for_status() 
        
        # Open the local file in binary write mode ('wb') and write the content
        with open(local_filename, 'wb') as f:
            f.write(response.content)
            
        print(f"Successfully downloaded '{file_path_in_repo}' as '{local_filename}'.")
        
    except requests.exceptions.RequestException as e:
        # Catch any request-related errors (e.g., network issues, invalid URL, 404 Not Found)
        print(f"Error downloading the file: {e}")
        print("Please double-check the GitHub username/organization, repository name, branch, and file path.")
        print(f"The URL attempted was: {raw_url}")

# --- Example Usage ---
# IMPORTANT: Replace these placeholders with the actual details of the file you want to download.

# 1. GitHub username or organization that owns the repository
GITHUB_USER = 'google' 

# 2. Name of the public repository
GITHUB_REPO = 'gemini-api-cookbook' 

# 3. Path to the file *within* the repository (e.g., 'notebooks/quickstarts/python/chat.ipynb')
#    Make sure this path is correct and includes the file extension.
FILE_PATH_IN_REPO = 'README.md'  

# 4. The branch where the file is located (commonly 'main' or 'master')
GITHUB_BRANCH = 'main'

# 5. Optional: The name you want to save the file as on your local machine.
#    If you leave this as None, the script will use the original filename from FILE_PATH_IN_REPO.
LOCAL_SAVE_NAME = 'gemini_cookbook_readme.md' 

# Call the function to start the download
download_github_file(
    user=GITHUB_USER, 
    repo=GITHUB_REPO, 
    file_path_in_repo=FILE_PATH_IN_REPO, 
    branch=GITHUB_BRANCH,
    local_filename=LOCAL_SAVE_NAME
)
