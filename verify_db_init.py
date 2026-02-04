import sys
import os
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import database

print(f"Current DB_DIR: {database.DB_DIR}")
try:
    client = database.get_client()
    print(f"Client initialized. Path: {client}")
    collection = database.get_collection()
    print(f"Collection 'papers' created or accessed. Count: {collection.count()}")
    
    if os.path.exists(database.DB_DIR):
        print(f"SUCCESS: Directory {database.DB_DIR} exists.")
    else:
        print(f"FAILURE: Directory {database.DB_DIR} does not exist.")
except Exception as e:
    print(f"ERROR: {e}")
