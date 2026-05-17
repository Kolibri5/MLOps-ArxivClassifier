import os

def test_data_directory_exists():
    """Memastikan folder data tersedia sebelum training dimulai"""
    assert os.path.exists("data"), "Folder 'data' tidak ditemukan!"
    
def test_src_directory_exists():
    """Memastikan folder source code tersedia"""
    assert os.path.exists("src"), "Folder 'src' tidak ditemukan!"