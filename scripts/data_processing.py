import pandas as pd

def clean_data(data):
    """Membersihkan data."""
    data.fillna({'Year': data['Year'].median()}, inplace=True)
    return data
