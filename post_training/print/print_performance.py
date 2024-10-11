import numpy as np
import util
import pandas as pd

class PrintPerformance:

    @staticmethod
    def in_sheet(objs): 
        field_names = _create_fields_sheet()
        sheet_values = _put_values_sheet(objs)
       
        df = pd.DataFrame(sheet_values, columns=field_names)
        df.to_excel(util.SHEET.PRINT_PATH, sheet_name=util.SHEET.NAME, index=False)
        
    
def _create_fields_sheet():
    list = []
    
    for database in util.DATABASES:
        for metr in util.TABLE.METRICS:
            item = f"{database} - {metr.upper()}(%)"
            list.append(item)  
            
    field_names = util.SHEET.FIELD_NAMES 
    field_names.extend(list)
    field_names.extend(util.SHEET.MEAN_NAMES)
    
    return field_names

def _put_values_sheet(objs):
    if util.USING_LATE_FUSION:
        specs_fusions = util.SPECTRUMS + [util.CHOSEN_FUSION] + util.LATE_FUSIONS
    else:
        specs_fusions = util.SPECTRUMS + [util.CHOSEN_FUSION]
        
    sorted_specs = {spec.upper(): i for i, spec in enumerate(specs_fusions)}
    sorted_objs = sorted(objs, key=lambda obj: (-ord(obj.model[0]), sorted_specs[obj.spectrum.upper()]))
    
    sheet_values = []
    size_metr = len(util.DATABASES)
       
    for j in range(0, len(sorted_objs), size_metr):
        name = f"{sorted_objs[j].spectrum.upper()}"
        data = [sorted_objs[j].model.upper(), name]
        
        metrs_list = []
        f1s_db = []
        ious_db = []
              
        for i in range(0, len(util.DATABASES)):
            f1s = sorted_objs[i+j].mean_f1s
            ious = sorted_objs[i+j].mean_ious
            
            metrs = [round(f1s*100, 2), round(ious*100, 2)]
            metrs_list.extend(metrs)
            
            f1s_db.append(f1s*100)
            ious_db.append(ious*100)
                 
        mean_f1s_db = round(np.mean(f1s_db), 2)
        mean_ious_db = round(np.mean(ious_db), 2)
        mean_metr_db = [mean_f1s_db, mean_ious_db]
        
        data.extend(metrs_list)
        data.extend(mean_metr_db)
        comma_data = [str(number).replace('.', ',') for number in data]
        sheet_values.append(comma_data)    
    
    return sheet_values    