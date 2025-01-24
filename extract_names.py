import os
import re

def extract_names(path):
    """
    Extracts file names from the given directory, removing extensions ".N" and ".O".
    
    Parameters:
        path (str): The directory path containing the files.
    
    Returns:
        list: A list of unique file names without extensions.
    """
    file_names = []
    
    # Vérifier si le dossier existe
    if os.path.exists(path) and os.path.isdir(path):
        # Parcourir tous les fichiers du dossier
        for file in os.listdir(path):
            # Retirer les extensions ".N" ou ".O" et ajouter à la liste
            name = re.sub(r"\.(N|O)$", "", file)
            file_names.append(name)
    
    # Supprimer les doublons éventuels
    return list(set(file_names))
