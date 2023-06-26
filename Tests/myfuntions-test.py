from EMGMOCAPproc import myfunctions

def test_crd2dict ():
    file_address = r'2023060907.c3d'
    assert myfunctions.crd2dict (file_address)