import numpy as np
import util
import pandas as pd

class PrintEceMetr:
    
    @staticmethod
    def in_sheet(objs):
        field_names = _create_fields_ece()
        sheet_values = _put_values_ece(objs)
    
        df = pd.DataFrame(sheet_values, columns=field_names)
        df.to_excel(util.SHEET.PRINT_PATH_ECE, sheet_name=util.SHEET.METRC_NAME, index=False)
        
def _create_fields_ece():
    list = []
    
    for database in util.DATABASES:
            item = f"{database} - ECE"
            list.append(item)  
            
    field_names = ["MODEL", "SPECTRUM"]
    field_names.extend(list)
    field_names.extend(["MEAN_ECE"])
    
    return field_names

def _put_values_ece(objs):
    if util.USING_LATE_FUSION:
        specs_fusions = util.SPECTRUMS + [util.CHOSEN_FUSION] + util.LATE_FUSIONS
    else:
        specs_fusions = util.SPECTRUMS + [util.CHOSEN_FUSION]
        
    sorted_specs = {spec.upper(): i for i, spec in enumerate(specs_fusions)}
    sorted_objs = sorted(objs, key=lambda obj: (-ord(obj.model[0]), sorted_specs[obj.spectrum.upper()]))
    
    sheet_values = []
    size_databases = len(util.DATABASES)
       
    for j in range(0, len(sorted_objs), size_databases):
        name = f"{sorted_objs[j].spectrum.upper()}"
        data = [sorted_objs[j].model.upper(), name]
        
        metrs_list = []
        ece_db = []
              
        for i in range(0, size_databases):
            eces = sorted_objs[i+j].mean_eces
            metrs = [round(eces, 2)]
            metrs_list.extend(metrs)
            ece_db.append(eces)
                 
        mean_ece_db = round(np.mean(ece_db), 2)
        mean_metr_db = [mean_ece_db]
        
        data.extend(metrs_list)
        data.extend(mean_metr_db)
        comma_data = [str(number).replace('.', ',') for number in data]
        sheet_values.append(comma_data)    
        
    return sheet_values
        