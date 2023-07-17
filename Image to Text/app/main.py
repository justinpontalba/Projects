# %%
import os
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
from db.db import create_table, insert_image_info, create_connection

def get_info_from_dicom(dicom_path):
    ds = dcmread(dicom_path)

    modality = str(ds.Modality)
    series_description = str(ds.SeriesDescription)
    slice_thickness = int(ds.SliceThickness)
    image_freq = float(ds.ImagingFrequency)
    patient_pos = str(ds.PatientPosition)

    info_dict = {
        "modality": modality,
        "series_description": series_description,
        "slice_thickness": slice_thickness,
        "image_freq": image_freq,
        "patient_pos": patient_pos
    }

    return info_dict

# %%
if __name__ == '__main__':

    # Define path to database
    database = r"D:\gitRepo\Projects\Projects\Image to Text\app\db\patients.db"
    images_path = r"D:\BRAIN_MRI\results\Neurohacking_data-0.0\BRAINIX\DICOM\T1\\"

    # create sql script to create table
    sql_create_images_table = """ CREATE TABLE IF NOT EXISTS patients (
                                        id integer PRIMARY KEY,
                                        image_path text,
                                        modality text,
                                        se_description text,
                                        slice_thickness real,
                                        image_freq real,
                                        patient_pos text
                                    ); """

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_images_table)
    else:
        print("Error! cannot create the database connection.")

    image_files = os.listdir(images_path)

    with conn:

        for id,image in enumerate(image_files):
            img_dict = get_info_from_dicom(images_path + image)

            patient = (id, 
            images_path + image, 
            img_dict['modality'],
            img_dict['series_description'], 
            img_dict['slice_thickness'], 
            img_dict['image_freq'], 
            img_dict['patient_pos']);

            patient_id = insert_image_info(conn, patient)
